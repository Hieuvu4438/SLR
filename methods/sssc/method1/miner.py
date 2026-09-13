from __future__ import annotations

import math
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .schemas import EditRecord, SCHEMA_VERSION
from .token_spans import (
    TokenizedCaption,
    auxiliary_edit_eligible,
    replace_occurrence,
)
from .utils import sha256_file, sha256_json, sha256_text


class MinerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OccurrenceDescriptor:
    occurrence_uid: str
    word: str
    text_uid: str
    video_uid: str
    group_uid: str
    vector: np.ndarray
    concentration: float
    peak_affinity: float
    support_entropy: float
    valid_clip_count: int


@dataclass(frozen=True)
class PrototypeRecord:
    word: str
    vector: np.ndarray
    total_occurrences: int
    retained_occurrences: int
    distinct_groups: int
    mean_concentration: float
    dispersion: float


@dataclass(frozen=True)
class MinedCaptionEdits:
    edits_by_text_uid: dict[str, tuple[EditRecord, ...]]
    rejection_counts: dict[str, int]


def occurrence_descriptor(
    *,
    occurrence_uid: str,
    word: str,
    text_uid: str,
    video_uid: str,
    group_uid: str,
    x_ref: np.ndarray | torch.Tensor,
    q: np.ndarray | torch.Tensor,
    video_valid: np.ndarray | torch.Tensor,
    tau: float,
) -> OccurrenceDescriptor:
    if tau <= 0:
        raise ValueError("support temperature must be positive")
    x = torch.as_tensor(x_ref, dtype=torch.float32)
    span = torch.as_tensor(q, dtype=torch.float32)
    valid = torch.as_tensor(video_valid, dtype=torch.bool)
    if x.ndim != 2 or span.shape != (x.shape[1],) or valid.shape != (x.shape[0],):
        raise MinerError("occurrence inputs must be x_ref[L,D], q[D], valid[L]")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(span).all()):
        raise MinerError("occurrence inputs contain NaN or infinity")
    valid_count = int(valid.sum())
    if valid_count < 2:
        raise MinerError("mining requires at least two valid reference clips")
    x = F.normalize(x, dim=-1, eps=1e-6)
    span = F.normalize(span, dim=-1, eps=1e-6)
    logits = torch.mv(x, span) / float(tau)
    support = logits.masked_fill(~valid, float("-inf")).softmax(dim=0)
    support = support.masked_fill(~valid, 0.0)
    pooled = torch.sum(support[:, None] * x, dim=0)
    if pooled.norm() <= 1e-6:
        raise MinerError("occurrence support pool is near zero")
    descriptor = F.normalize(pooled, dim=0, eps=1e-6)
    probabilities = support[valid]
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12))
    normalized_entropy = entropy / math.log(valid_count)
    concentration = 1.0 - normalized_entropy
    peak_affinity = torch.mv(x[valid], span).max()
    values = (descriptor, concentration, peak_affinity, normalized_entropy)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise MinerError("occurrence descriptor is nonfinite")
    return OccurrenceDescriptor(
        occurrence_uid=occurrence_uid,
        word=word,
        text_uid=text_uid,
        video_uid=video_uid,
        group_uid=group_uid,
        vector=descriptor.cpu().numpy().astype(np.float32, copy=False),
        concentration=float(concentration),
        peak_affinity=float(peak_affinity),
        support_entropy=float(normalized_entropy),
        valid_clip_count=valid_count,
    )


def _normalized_mean(vectors: Sequence[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(vectors).astype(np.float32), axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1e-6:
        raise MinerError("prototype mean is zero or nonfinite")
    return (mean / norm).astype(np.float32, copy=False)


def fit_visual_prototypes(
    descriptors: Sequence[OccurrenceDescriptor],
    *,
    retained_fraction: float,
    minimum_concentration: float,
    minimum_retained_occurrences: int,
    minimum_distinct_groups: int,
) -> tuple[dict[str, PrototypeRecord], dict[str, dict[str, Any]]]:
    if not 0 < retained_fraction <= 1:
        raise ValueError("retained_fraction must be in (0,1]")
    if minimum_retained_occurrences < 1 or minimum_distinct_groups < 1:
        raise ValueError("prototype count thresholds must be positive")
    by_word: dict[str, list[OccurrenceDescriptor]] = defaultdict(list)
    for descriptor in descriptors:
        if descriptor.vector.ndim != 1 or not np.isfinite(descriptor.vector).all():
            raise MinerError(f"invalid descriptor vector: {descriptor.occurrence_uid}")
        by_word[descriptor.word].append(descriptor)
    prototypes: dict[str, PrototypeRecord] = {}
    reports: dict[str, dict[str, Any]] = {}
    for word in sorted(by_word):
        occurrences = sorted(
            by_word[word],
            key=lambda item: (-item.concentration, item.occurrence_uid, item.video_uid),
        )
        retain_count = math.ceil(len(occurrences) * retained_fraction)
        retained = [
            item
            for item in occurrences[:retain_count]
            if item.concentration > minimum_concentration
        ]
        groups: dict[str, list[OccurrenceDescriptor]] = defaultdict(list)
        for item in retained:
            groups[item.group_uid].append(item)
        report = {
            "total_occurrences": len(occurrences),
            "top_fraction_count": retain_count,
            "retained_occurrences": len(retained),
            "distinct_groups": len(groups),
            "rejected_low_concentration": retain_count - len(retained),
            "eligible": False,
        }
        if len(retained) < minimum_retained_occurrences:
            report["rejection_reason"] = "insufficient_retained_occurrences"
            reports[word] = report
            continue
        if len(groups) < minimum_distinct_groups:
            report["rejection_reason"] = "insufficient_distinct_groups"
            reports[word] = report
            continue
        group_means = [_normalized_mean([item.vector for item in groups[group]]) for group in sorted(groups)]
        prototype = _normalized_mean(group_means)
        descriptor_matrix = np.stack([item.vector for item in retained]).astype(np.float32)
        dispersion = float(np.mean(1.0 - descriptor_matrix @ prototype))
        record = PrototypeRecord(
            word=word,
            vector=prototype,
            total_occurrences=len(occurrences),
            retained_occurrences=len(retained),
            distinct_groups=len(groups),
            mean_concentration=float(np.mean([item.concentration for item in retained])),
            dispersion=dispersion,
        )
        prototypes[word] = record
        report.update(
            {
                "eligible": True,
                "mean_concentration": record.mean_concentration,
                "dispersion": record.dispersion,
            }
        )
        reports[word] = report
    return prototypes, reports


def build_candidate_graph(
    prototypes: Mapping[str, PrototypeRecord],
    *,
    minimum_cosine: float,
    candidates_per_word: int,
    block_size: int = 256,
) -> dict[str, tuple[tuple[str, float], ...]]:
    if not -1 <= minimum_cosine <= 1 or candidates_per_word < 1 or block_size < 1:
        raise ValueError("invalid candidate graph thresholds")
    words = sorted(prototypes)
    if not words:
        return {}
    matrix = np.stack([prototypes[word].vector for word in words]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or not np.allclose(norms, 1.0, atol=2e-5):
        raise MinerError("prototype vectors must be finite and normalized")
    graph: dict[str, tuple[tuple[str, float], ...]] = {}
    for start in range(0, len(words), block_size):
        similarities = matrix[start : start + block_size] @ matrix.T
        for local_row, word in enumerate(words[start : start + block_size]):
            global_row = start + local_row
            candidates = [
                (candidate, float(similarities[local_row, column]))
                for column, candidate in enumerate(words)
                if column != global_row and similarities[local_row, column] >= minimum_cosine
            ]
            candidates.sort(key=lambda item: (-item[1], item[0]))
            graph[word] = tuple(candidates[:candidates_per_word])
    return graph


def construct_caption_edits(
    captions: Mapping[str, TokenizedCaption],
    candidate_graph: Mapping[str, Sequence[tuple[str, float]]],
    *,
    tokenizer: Any,
    original_train_caption_hashes: set[str],
    max_edits_per_caption: int | None,
    miner_version: str,
    max_positions: int = 32,
) -> MinedCaptionEdits:
    if max_edits_per_caption is not None and max_edits_per_caption < 1:
        raise ValueError("max_edits_per_caption must be positive")
    edits_by_text: dict[str, tuple[EditRecord, ...]] = {}
    rejection_counts: Counter[str] = Counter()
    for text_uid in sorted(captions):
        caption = captions[text_uid]
        occurrences = sorted(caption.lexical_spans, key=lambda item: item.char_span)
        lexical_units = {item.surface for item in occurrences}
        accepted: list[EditRecord] = []
        seen_negative_hashes: set[str] = set()
        maximum_rank = max((len(candidate_graph.get(item.surface, ())) for item in occurrences), default=0)
        for candidate_rank in range(maximum_rank):
            for occurrence in occurrences:
                candidates = candidate_graph.get(occurrence.surface, ())
                if candidate_rank >= len(candidates):
                    continue
                replacement, cosine = candidates[candidate_rank]
                if replacement == occurrence.surface:
                    rejection_counts["same_word"] += 1
                    continue
                if replacement in lexical_units:
                    rejection_counts["replacement_already_in_caption"] += 1
                    continue
                try:
                    negative, negative_span = replace_occurrence(
                        caption,
                        occurrence,
                        replacement,
                        tokenizer=tokenizer,
                        max_positions=max_positions,
                    )
                except ValueError:
                    rejection_counts["replacement_lexical_or_tokenization_failure"] += 1
                    continue
                eligible, reason = auxiliary_edit_eligible(
                    caption, occurrence, negative, negative_span
                )
                if not eligible:
                    rejection_counts[reason or "ineligible"] += 1
                    continue
                if negative.caption_hash in original_train_caption_hashes:
                    rejection_counts["matches_original_train_caption"] += 1
                    continue
                if negative.caption_hash in seen_negative_hashes:
                    rejection_counts["duplicate_edited_caption"] += 1
                    continue
                seen_negative_hashes.add(negative.caption_hash)
                edit_uid = "edit:" + sha256_text(
                    "\x1f".join(
                        (text_uid, occurrence.occurrence_uid, replacement, negative.caption_hash, miner_version)
                    )
                )
                record = EditRecord(
                    schema_version=SCHEMA_VERSION,
                    edit_uid=edit_uid,
                    text_uid=text_uid,
                    positive_caption_hash=caption.caption_hash,
                    negative_canonical_text=negative.canonical_text,
                    negative_caption_hash=negative.caption_hash,
                    occurrence_uid=occurrence.occurrence_uid,
                    source_word=occurrence.surface,
                    replacement_word=replacement,
                    positive_char_span=occurrence.char_span,
                    negative_char_span=negative_span.char_span,
                    positive_token_positions=occurrence.token_positions,
                    negative_token_positions=negative_span.token_positions,
                    positive_full_bpe_count=len(caption.full_bpe_ids),
                    negative_full_bpe_count=len(negative.full_bpe_ids),
                    prototype_cosine=float(cosine),
                    eligible=True,
                    rejection_reason=None,
                    miner_version=miner_version,
                )
                record.validate()
                accepted.append(record)
                if max_edits_per_caption is not None and len(accepted) >= max_edits_per_caption:
                    break
            if max_edits_per_caption is not None and len(accepted) >= max_edits_per_caption:
                break
        if not accepted:
            rejection_counts["caption_without_valid_edit"] += 1
        edits_by_text[text_uid] = tuple(accepted)
    return MinedCaptionEdits(edits_by_text, dict(sorted(rejection_counts.items())))


def write_mining_artifacts(
    cache_root: str | Path,
    *,
    descriptors: Sequence[OccurrenceDescriptor],
    prototypes: Mapping[str, PrototypeRecord],
    prototype_report: Mapping[str, Mapping[str, Any]],
    candidate_graph: Mapping[str, Sequence[tuple[str, float]]],
    mined: MinedCaptionEdits,
    resource_hashes: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(cache_root) / "mining"
    if destination.exists():
        raise MinerError(f"refusing to overwrite mining artifacts: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building.", dir=destination.parent)
    )
    try:
        descriptor_vectors = (
            np.stack([item.vector for item in descriptors]).astype(np.float32)
            if descriptors
            else np.empty((0, 0), dtype=np.float32)
        )
        words = sorted(prototypes)
        prototype_vectors = (
            np.stack([prototypes[word].vector for word in words]).astype(np.float32)
            if words
            else np.empty((0, descriptor_vectors.shape[1] if descriptor_vectors.ndim == 2 else 0), dtype=np.float32)
        )
        np.save(staging / "occurrence_descriptors.npy", descriptor_vectors, allow_pickle=False)
        np.save(staging / "visual_prototypes.npy", prototype_vectors, allow_pickle=False)
        with (staging / "occurrences.jsonl").open("w", encoding="utf-8") as handle:
            for row, item in enumerate(descriptors):
                record = asdict(item)
                record.pop("vector")
                record["row"] = row
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        with (staging / "prototypes.jsonl").open("w", encoding="utf-8") as handle:
            for row, word in enumerate(words):
                record = asdict(prototypes[word])
                record.pop("vector")
                record["row"] = row
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        with (staging / "edits.jsonl").open("w", encoding="utf-8") as handle:
            edit_count = 0
            for text_uid in sorted(mined.edits_by_text_uid):
                for caption_order, edit in enumerate(mined.edits_by_text_uid[text_uid]):
                    record = asdict(edit)
                    record["caption_order"] = caption_order
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    edit_count += 1
        graph_payload = {
            word: [{"replacement": candidate, "cosine": cosine} for candidate, cosine in values]
            for word, values in sorted(candidate_graph.items())
        }
        (staging / "candidate_graph.json").write_text(
            json.dumps(graph_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        files = {
            name: sha256_file(staging / name)
            for name in (
                "occurrence_descriptors.npy",
                "visual_prototypes.npy",
                "occurrences.jsonl",
                "prototypes.jsonl",
                "edits.jsonl",
                "candidate_graph.json",
            )
        }
        report = {
            "schema_version": 1,
            "status": "complete",
            "resource_hashes": dict(resource_hashes),
            "descriptor_count": len(descriptors),
            "prototype_count": len(prototypes),
            "candidate_edge_count": sum(len(values) for values in candidate_graph.values()),
            "caption_count": len(mined.edits_by_text_uid),
            "captions_with_edits": sum(bool(values) for values in mined.edits_by_text_uid.values()),
            "edit_count": edit_count,
            "rejection_counts": mined.rejection_counts,
            "prototype_report": dict(prototype_report),
            "diagnostics": dict(diagnostics or {}),
            "files": files,
        }
        report["content_sha256"] = sha256_json(report)
        (staging / "mining_report.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
