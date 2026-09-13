from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Method1Config
from .data import _load_records
from .manifests import iter_jsonl
from .miner import (
    MinedCaptionEdits,
    MinerError,
    build_candidate_graph,
    construct_caption_edits,
    fit_visual_prototypes,
    occurrence_descriptor,
    write_mining_artifacts,
)
from .model_factory import build_upret_model, load_exact_student_state
from .reference import (
    encode_reference_text,
    freeze_reference,
    pool_reference_spans,
    validate_reference_cache,
    write_negative_span_cache,
)
from .reference_pipeline import reference_cache_identity
from .schemas import GroupRecord, TextRecord
from .token_spans import tokenize_with_spans
from .upstream import create_upret_tokenizer
from .utils import sha256_file


def _index_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records = list(iter_jsonl(path))
    output = {str(record[key]): record for record in records}
    if len(output) != len(records):
        raise MinerError(f"duplicate {key} in {path}")
    return output


def _encode_negative_spans(
    reference: torch.nn.Module,
    tokenizer: Any,
    edits,
    *,
    max_positions: int,
    batch_size: int,
    device: str | torch.device,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for start in range(0, len(edits), batch_size):
        current = edits[start : start + batch_size]
        encoded = [
            tokenize_with_spans(
                edit.negative_canonical_text,
                text_uid=edit.text_uid,
                tokenizer=tokenizer,
                max_positions=max_positions,
            )
            for edit in current
        ]
        for edit, caption in zip(current, encoded, strict=True):
            if caption.caption_hash != edit.negative_caption_hash:
                raise MinerError(f"negative caption hash changed: {edit.edit_uid}")
        ids = torch.tensor([item.input_ids for item in encoded], dtype=torch.long, device=device)
        valid = torch.tensor([item.text_valid for item in encoded], dtype=torch.bool, device=device)
        tokens, _ = encode_reference_text(reference, ids, torch.zeros_like(ids), valid)
        vectors = pool_reference_spans(
            reference, tokens, [edit.negative_token_positions for edit in current]
        ).cpu().numpy()
        for edit, vector in zip(current, vectors, strict=True):
            output[edit.edit_uid] = vector.astype(np.float32, copy=False)
    return output


def mine_reference_negatives(
    config: Method1Config,
    *,
    upret_root: str | Path = "third_party/UPRet",
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
) -> dict[str, Any]:
    cache_root = Path(config.reference.cache_dir)
    identity = reference_cache_identity(config)
    arrays = validate_reference_cache(cache_root, expected_identity=identity)
    video_index = _index_by(cache_root / "video_index.jsonl", "video_uid")
    positive_index = list(iter_jsonl(cache_root / "positive_span_index.jsonl"))
    groups = _load_records(Path(config.data.manifest_dir) / "groups.jsonl", GroupRecord)
    texts = _load_records(Path(config.data.manifest_dir) / "texts.jsonl", TextRecord)
    train_groups = {record.text_uid: record for record in groups if ":train:" in record.group_uid}
    train_texts = {record.text_uid: record for record in texts if record.split == "train"}
    if set(train_texts) != set(train_groups):
        raise MinerError("train text/group manifests do not have a one-to-one identity mapping")

    descriptors = []
    descriptor_rejections: Counter[str] = Counter()
    for occurrence in positive_index:
        text_uid = str(occurrence["text_uid"])
        group = train_groups.get(text_uid)
        if group is None:
            raise MinerError(f"positive occurrence is outside train groups: {text_uid}")
        positive_row = int(occurrence["row"])
        q = arrays["reference_positive_spans"][positive_row]
        for video_uid in group.video_uids:
            if video_uid not in video_index:
                raise MinerError(f"group video is absent from reference cache: {video_uid}")
            video_row = int(video_index[video_uid]["row"])
            try:
                descriptors.append(
                    occurrence_descriptor(
                        occurrence_uid=str(occurrence["occurrence_uid"]),
                        word=str(occurrence["word"]),
                        text_uid=text_uid,
                        video_uid=video_uid,
                        group_uid=group.group_uid,
                        x_ref=arrays["reference_video_tokens"][video_row],
                        q=q,
                        video_valid=arrays["reference_video_valid"][video_row],
                        tau=config.auxiliary.tau_support,
                    )
                )
            except MinerError as error:
                descriptor_rejections[str(error)] += 1
    prototypes, prototype_report = fit_visual_prototypes(
        descriptors,
        retained_fraction=config.miner.retained_occurrence_fraction,
        minimum_concentration=config.miner.minimum_concentration,
        minimum_retained_occurrences=config.miner.minimum_retained_occurrences,
        minimum_distinct_groups=config.miner.minimum_distinct_groups,
    )
    graph = build_candidate_graph(
        prototypes,
        minimum_cosine=config.miner.candidate_cosine_minimum,
        candidates_per_word=config.miner.candidates_per_word,
    )
    tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
    captions = {
        text_uid: tokenize_with_spans(
            text.raw_text,
            text_uid=text_uid,
            tokenizer=tokenizer,
            max_positions=config.data.text_max_positions,
        )
        for text_uid, text in train_texts.items()
    }
    mined_all = construct_caption_edits(
        captions,
        graph,
        tokenizer=tokenizer,
        original_train_caption_hashes={text.caption_hash for text in train_texts.values()},
        max_edits_per_caption=None,
        miner_version=config.miner.version,
        max_positions=config.data.text_max_positions,
    )
    all_edits = [
        edit
        for text_uid in sorted(mined_all.edits_by_text_uid)
        for edit in mined_all.edits_by_text_uid[text_uid]
    ]
    if not all_edits:
        raise MinerError("mining produced no token-valid candidate edits")
    model, _ = build_upret_model(config, upret_root=upret_root)
    checkpoint = load_exact_student_state(model, config.reference.checkpoint)
    if checkpoint.get("dev_selection") is None or checkpoint.get("arm") != "base_initial":
        raise MinerError("negative spans require the same dev-selected base reference")
    reference = freeze_reference(model.to(device))
    batch_size = int(batch_size or config.evaluation.encode_batch_size)
    negative_vectors = _encode_negative_spans(
        reference,
        tokenizer,
        all_edits,
        max_positions=config.data.text_max_positions,
        batch_size=batch_size,
        device=device,
    )
    positive_rows = {
        str(record["occurrence_uid"]): int(record["row"]) for record in positive_index
    }
    filtered: dict[str, tuple] = {}
    rejection_counts = Counter(mined_all.rejection_counts)
    for text_uid in sorted(mined_all.edits_by_text_uid):
        accepted = []
        for edit in mined_all.edits_by_text_uid[text_uid]:
            positive = arrays["reference_positive_spans"][positive_rows[edit.occurrence_uid]]
            negative = negative_vectors[edit.edit_uid]
            difference = np.linalg.norm(positive - negative)
            if not np.isfinite(difference) or difference <= 1e-6:
                rejection_counts["near_zero_normalized_text_difference"] += 1
                continue
            if len(accepted) < config.miner.cached_negatives_per_caption:
                accepted.append(edit)
            else:
                rejection_counts["beyond_caption_cache_limit"] += 1
        filtered[text_uid] = tuple(accepted)
    mined = MinedCaptionEdits(filtered, dict(sorted(rejection_counts.items())))
    retained_edits = [
        edit for text_uid in sorted(filtered) for edit in filtered[text_uid]
    ]
    if not retained_edits:
        raise MinerError("all mined edits failed frozen text-difference validation")
    cache_meta = json.loads((cache_root / "cache_meta.json").read_text(encoding="utf-8"))
    resource_hashes = {
        "reference_identity_sha256": identity.digest,
        "reference_cache_content_sha256": cache_meta["content_sha256"],
        "teacher_checkpoint_sha256": identity.teacher_checkpoint_sha256,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "train_text_manifest_sha256": sha256_file(
            Path(config.data.manifest_dir) / "texts.jsonl"
        ),
        "train_group_manifest_sha256": sha256_file(
            Path(config.data.manifest_dir) / "groups.jsonl"
        ),
        "miner_config": {
            "version": config.miner.version,
            "retained_occurrence_fraction": config.miner.retained_occurrence_fraction,
            "minimum_concentration": config.miner.minimum_concentration,
            "minimum_retained_occurrences": config.miner.minimum_retained_occurrences,
            "minimum_distinct_groups": config.miner.minimum_distinct_groups,
            "candidate_cosine_minimum": config.miner.candidate_cosine_minimum,
            "candidates_per_word": config.miner.candidates_per_word,
            "cached_negatives_per_caption": config.miner.cached_negatives_per_caption,
        },
        "descriptor_rejections": dict(sorted(descriptor_rejections.items())),
    }
    bundle_destination = cache_root / "auxiliary"
    if bundle_destination.exists():
        raise MinerError(f"refusing to overwrite auxiliary cache bundle: {bundle_destination}")
    bundle_staging = Path(
        tempfile.mkdtemp(prefix=".auxiliary.building.", dir=cache_root)
    )
    try:
        mining_report = write_mining_artifacts(
            bundle_staging,
            descriptors=descriptors,
            prototypes=prototypes,
            prototype_report=prototype_report,
            candidate_graph=graph,
            mined=mined,
            resource_hashes=resource_hashes,
        )
        ordered_edit_uids = [edit.edit_uid for edit in retained_edits]
        negative_array = np.stack(
            [negative_vectors[uid] for uid in ordered_edit_uids]
        ).astype(np.float32)
        negative_report = write_negative_span_cache(
            bundle_staging,
            reference_identity_sha256=identity.digest,
            mining_content_sha256=mining_report["content_sha256"],
            vectors=negative_array,
            edit_uids=ordered_edit_uids,
        )
        os.replace(bundle_staging, bundle_destination)
    except BaseException:
        shutil.rmtree(bundle_staging, ignore_errors=True)
        raise
    return {
        "schema_version": 1,
        "status": "complete",
        "descriptor_count": len(descriptors),
        "prototype_count": len(prototypes),
        "edit_count": len(retained_edits),
        "captions_with_edits": sum(bool(values) for values in filtered.values()),
        "mining_content_sha256": mining_report["content_sha256"],
        "negative_cache_content_sha256": negative_report["content_sha256"],
    }
