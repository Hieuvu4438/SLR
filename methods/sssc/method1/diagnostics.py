from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .auxiliary_cache import Method1AuxiliaryCache
from .config import Method1Config
from .data import _load_records
from .losses.shared_support import reference_support
from .miner import construct_caption_edits
from .model_factory import build_upret_model, load_exact_student_state
from .reference import (
    encode_reference_text,
    encode_reference_video,
    freeze_reference,
    pool_reference_spans,
)
from .reference_pipeline import _video_batch, reference_cache_identity
from .schemas import GroupRecord, TextRecord, VideoRecord
from .token_spans import TokenizedCaption, tokenize_with_spans
from .upstream import create_upret_tokenizer
from .utils import atomic_json_dump, atomic_jsonl_dump, sha256_file


class DiagnosticError(RuntimeError):
    pass


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _load_frozen_candidate_graph(cache_root: Path) -> dict[str, tuple[tuple[str, float], ...]]:
    path = cache_root / "auxiliary" / "mining" / "candidate_graph.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"cannot read frozen train candidate graph: {error}") from error
    output = {}
    for word, candidates in raw.items():
        output[str(word)] = tuple(
            (str(item["replacement"]), float(item["cosine"])) for item in candidates
        )
    return output


def _encode_span_vectors(
    reference: torch.nn.Module,
    tokenizer: Any,
    captions: Sequence[tuple[str, str, tuple[int, ...]]],
    *,
    max_positions: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for start in range(0, len(captions), batch_size):
        current = captions[start : start + batch_size]
        encoded = [
            tokenize_with_spans(
                text,
                text_uid=text_uid,
                tokenizer=tokenizer,
                max_positions=max_positions,
            )
            for text_uid, text, _ in current
        ]
        ids = torch.tensor([item.input_ids for item in encoded], dtype=torch.long, device=device)
        valid = torch.tensor([item.text_valid for item in encoded], dtype=torch.bool, device=device)
        tokens, _ = encode_reference_text(reference, ids, torch.zeros_like(ids), valid)
        vectors = pool_reference_spans(
            reference, tokens, [positions for _, _, positions in current]
        ).cpu().numpy()
        output.extend(vector.astype(np.float32, copy=False) for vector in vectors)
    return output


def _support_metrics(
    x_ref: np.ndarray,
    video_valid: np.ndarray,
    selected_feature_indices: Sequence[int],
    q_positive: np.ndarray,
    q_negative: np.ndarray,
    *,
    tau: float,
    margin: float,
) -> dict[str, Any]:
    x = torch.from_numpy(np.asarray(x_ref, dtype=np.float32))[None]
    valid = torch.from_numpy(np.asarray(video_valid, dtype=np.bool_))[None]
    positive = torch.from_numpy(np.asarray(q_positive, dtype=np.float32))[None, None, None]
    negative = torch.from_numpy(np.asarray(q_negative, dtype=np.float32))[None, None, None]
    positive_support = reference_support(x, positive, valid, tau)[0, 0, 0]
    negative_support = reference_support(x, negative, valid, tau)[0, 0, 0]
    midpoint = 0.5 * (positive_support + negative_support)
    jsd = 0.5 * (
        torch.sum(
            torch.where(
                positive_support > 0,
                positive_support * torch.log(positive_support / midpoint.clamp_min(1e-12)),
                torch.zeros_like(positive_support),
            )
        )
        + torch.sum(
            torch.where(
                negative_support > 0,
                negative_support * torch.log(negative_support / midpoint.clamp_min(1e-12)),
                torch.zeros_like(negative_support),
            )
        )
    )
    valid_count = int(valid.sum())
    positive_entropy = -torch.sum(
        positive_support[valid[0]] * torch.log(positive_support[valid[0]] + 1e-12)
    )
    normalized_entropy = (
        float(positive_entropy / math.log(valid_count)) if valid_count > 1 else 0.0
    )
    positive_peak = int(torch.argmax(positive_support))
    negative_peak = int(torch.argmax(negative_support))
    selected = np.asarray(selected_feature_indices, dtype=np.int64)
    shared_delta = torch.einsum(
        "l,ld,d->", positive_support, x[0], positive[0, 0, 0] - negative[0, 0, 0]
    )
    independent_delta = torch.einsum(
        "l,ld,d->", positive_support, x[0], positive[0, 0, 0]
    ) - torch.einsum("l,ld,d->", negative_support, x[0], negative[0, 0, 0])
    return {
        "q_difference_norm": float(np.linalg.norm(q_positive - q_negative)),
        "text_vector_cosine": float(np.dot(q_positive, q_negative)),
        "support_overlap": float(torch.minimum(positive_support, negative_support).sum()),
        "support_jsd": float(jsd),
        "positive_support_entropy": normalized_entropy,
        "positive_support_concentration": 1.0 - normalized_entropy,
        "positive_peak_feature_index": int(selected[positive_peak]),
        "negative_peak_feature_index": int(selected[negative_peak]),
        "peak_displacement_feature_rows": int(
            abs(selected[positive_peak] - selected[negative_peak])
        ),
        "shared_margin": float(shared_delta),
        "independent_margin": float(independent_delta),
        "shared_hinge_active": bool(shared_delta < margin),
        "independent_hinge_active": bool(independent_delta < margin),
        "valid_clip_count": valid_count,
    }


def support_diagnostics(
    config: Method1Config,
    *,
    split: str,
    upret_root: str | Path = "third_party/UPRet",
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    if split != "dev":
        raise DiagnosticError("support diagnostics are dev-only before configuration lock")
    cache_root = Path(config.reference.cache_dir)
    identity = reference_cache_identity(config)
    # This performs the full train-cache/miner/negative-cache identity audit without reusing
    # any train example as a dev observation.
    Method1AuxiliaryCache(cache_root, expected_identity=identity)
    graph = _load_frozen_candidate_graph(cache_root)
    model, _ = build_upret_model(config, upret_root=upret_root)
    checkpoint = load_exact_student_state(model, config.reference.checkpoint)
    if (
        checkpoint.get("arm") != "base_initial"
        or checkpoint.get("dev_selection") is None
        or not checkpoint.get("training_run_complete", False)
    ):
        raise DiagnosticError("diagnostics require the completed selected baseline reference")
    torch_device = torch.device(device)
    reference = freeze_reference(model.to(torch_device))
    tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
    manifest_root = Path(config.data.manifest_dir)
    texts = {
        record.text_uid: record
        for record in _load_records(manifest_root / "texts.jsonl", TextRecord)
    }
    groups = [
        record
        for record in _load_records(manifest_root / "groups.jsonl", GroupRecord)
        if record.group_uid.startswith(f"{config.data.dataset}:{split}:")
    ]
    videos = {
        record.video_uid: record
        for record in _load_records(manifest_root / "videos.jsonl", VideoRecord)
        if record.split == split
    }
    train_hashes = {record.caption_hash for record in texts.values() if record.split == "train"}
    tokenized: dict[str, TokenizedCaption] = {
        group.text_uid: tokenize_with_spans(
            texts[group.text_uid].raw_text,
            text_uid=group.text_uid,
            tokenizer=tokenizer,
            max_positions=config.data.text_max_positions,
        )
        for group in groups
    }
    mined = construct_caption_edits(
        tokenized,
        graph,
        tokenizer=tokenizer,
        original_train_caption_hashes=train_hashes,
        max_edits_per_caption=config.miner.cached_negatives_per_caption,
        miner_version=f"{config.miner.version}:dev_diagnostic_only",
        max_positions=config.data.text_max_positions,
    )
    edits = [
        edit
        for text_uid in sorted(mined.edits_by_text_uid)
        for edit in mined.edits_by_text_uid[text_uid]
    ]
    if not edits:
        raise DiagnosticError("frozen train vocabulary produced no eligible dev diagnostic edits")
    text_inputs = []
    for edit in edits:
        positive = tokenized[edit.text_uid]
        text_inputs.append(
            (edit.text_uid, positive.canonical_text, edit.positive_token_positions)
        )
        text_inputs.append(
            (edit.text_uid, edit.negative_canonical_text, edit.negative_token_positions)
        )
    vectors = _encode_span_vectors(
        reference,
        tokenizer,
        text_inputs,
        max_positions=config.data.text_max_positions,
        batch_size=config.evaluation.encode_batch_size,
        device=torch_device,
    )
    edit_vectors = {
        edit.edit_uid: (vectors[2 * index], vectors[2 * index + 1])
        for index, edit in enumerate(edits)
    }
    per_example: list[dict[str, Any]] = []
    edits_by_text = mined.edits_by_text_uid
    for group in groups:
        group_edits = edits_by_text.get(group.text_uid, ())
        if not group_edits:
            continue
        group_videos = [videos[video_uid] for video_uid in group.video_uids]
        for start in range(0, len(group_videos), config.evaluation.encode_batch_size):
            current = group_videos[start : start + config.evaluation.encode_batch_size]
            batch, selected = _video_batch(config, current)
            batch = {key: value.to(torch_device) for key, value in batch.items()}
            encoded_video, valid = encode_reference_video(reference, batch)
            for row, (video_record, selected_indexes) in enumerate(
                zip(current, selected, strict=True)
            ):
                for edit in group_edits:
                    q_positive, q_negative = edit_vectors[edit.edit_uid]
                    metrics = _support_metrics(
                        encoded_video[row].cpu().numpy(),
                        valid[row].cpu().numpy(),
                        selected_indexes,
                        q_positive,
                        q_negative,
                        tau=config.auxiliary.tau_support,
                        margin=config.auxiliary.margin,
                    )
                    per_example.append(
                        {
                            "schema_version": 1,
                            "split": split,
                            "diagnostic_only": True,
                            "semantic_validity": "unknown",
                            "video_uid": video_record.video_uid,
                            "group_uid": group.group_uid,
                            "text_uid": group.text_uid,
                            "edit_uid": edit.edit_uid,
                            "occurrence_uid": edit.occurrence_uid,
                            "source_word": edit.source_word,
                            "replacement_word": edit.replacement_word,
                            "prototype_cosine": edit.prototype_cosine,
                            **metrics,
                        }
                    )
    fields = (
        "q_difference_norm",
        "text_vector_cosine",
        "support_overlap",
        "support_jsd",
        "positive_support_entropy",
        "positive_support_concentration",
        "peak_displacement_feature_rows",
        "shared_margin",
        "independent_margin",
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "kind": "support",
        "split": split,
        "diagnostic_only": True,
        "fit_split": "train",
        "observation_split": split,
        "semantic_accuracy_computed": False,
        "reference_identity_sha256": identity.digest,
        "candidate_graph_sha256": sha256_file(
            cache_root / "auxiliary" / "mining" / "candidate_graph.json"
        ),
        "caption_count": len(groups),
        "caption_with_edit_count": sum(bool(value) for value in edits_by_text.values()),
        "candidate_edit_count": len(edits),
        "video_edit_observation_count": len(per_example),
        "construction_rejections": mined.rejection_counts,
        "shared_active_fraction": float(
            np.mean([item["shared_hinge_active"] for item in per_example])
        ),
        "independent_active_fraction": float(
            np.mean([item["independent_hinge_active"] for item in per_example])
        ),
        "distributions": {
            field: _distribution([float(item[field]) for item in per_example])
            for field in fields
        },
    }
    output_root = Path(config.output.root) / "diagnostics"
    records_path = output_root / f"{split}_support_records.jsonl"
    summary_path = output_root / f"{split}_support_diagnostics.json"
    atomic_jsonl_dump(per_example, records_path)
    atomic_json_dump(summary, summary_path)
    summary["records_path"] = str(records_path.resolve())
    summary["summary_path"] = str(summary_path.resolve())
    return summary
