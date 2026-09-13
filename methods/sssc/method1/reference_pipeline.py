from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Method1Config
from .data import _load_feature, _load_records
from .model_factory import build_upret_model, load_exact_student_state
from .reference import (
    ReferenceCacheIdentity,
    encode_reference_text,
    encode_reference_video,
    freeze_reference,
    pool_reference_spans,
    write_reference_cache,
)
from .sampling import mix_and_sample_features
from .schemas import TextRecord, VideoRecord
from .token_spans import tokenize_with_spans
from .upstream import create_upret_tokenizer
from .utils import sha256_file, sha256_json


REFERENCE_IMPLEMENTATION_VERSION = "reference_cache_v1"


def reference_cache_identity(
    config: Method1Config,
    *,
    teacher_checkpoint: str | Path | None = None,
) -> ReferenceCacheIdentity:
    root = Path(config.data.manifest_dir)
    checkpoint = Path(teacher_checkpoint or config.reference.checkpoint)
    sampler = {
        "feature_sampling": config.data.feature_sampling,
        "feature_len": config.data.feature_len,
        "feature_dim": config.data.feature_dim,
        "combine_type": config.data.combine_type,
        "agnostic_weight": config.data.agnostic_weight,
        "algorithm": "legacy_linspace_mix_v1",
    }
    return ReferenceCacheIdentity(
        teacher_checkpoint_sha256=sha256_file(checkpoint),
        tokenizer_sha256=sha256_file(config.model.bpe_path),
        caption_manifest_sha256=sha256_file(root / "texts.jsonl"),
        feature_manifest_sha256=sha256_file(root / "videos.jsonl"),
        mixture_sampler_sha256=sha256_json(sampler),
        implementation_version=REFERENCE_IMPLEMENTATION_VERSION,
        cache_dtype=config.reference.cache_dtype,
    )


def _video_batch(config: Method1Config, records: list[VideoRecord]) -> tuple[dict[str, Any], list[np.ndarray]]:
    features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    selected_indexes: list[np.ndarray] = []
    for record in records:
        agnostic = _load_feature(record.agnostic_path)
        aware = _load_feature(record.aware_path)
        mixed, valid, selected = mix_and_sample_features(
            agnostic,
            aware,
            agnostic_weight=config.data.agnostic_weight,
            feature_len=config.data.feature_len,
        )
        video_ignore = np.ones(config.data.feature_len + 1, dtype=np.bool_)
        video_ignore[1:] = ~valid
        features.append(torch.from_numpy(mixed.T[:, :, None].copy()))
        masks.append(torch.from_numpy(video_ignore))
        selected_indexes.append(selected)
    return {
        "video_features": torch.stack(features),
        "video_ignore_raw": torch.stack(masks),
    }, selected_indexes


def create_reference_cache(
    config: Method1Config,
    *,
    upret_root: str | Path = "third_party/UPRet",
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    checkpoint_path = Path(config.reference.checkpoint)
    identity = reference_cache_identity(config, teacher_checkpoint=checkpoint_path)
    model, _ = build_upret_model(config, upret_root=upret_root)
    checkpoint = load_exact_student_state(model, checkpoint_path)
    if (
        checkpoint.get("dev_selection") is None
        or checkpoint.get("arm") != "base_initial"
        or not checkpoint.get("training_run_complete", False)
    ):
        raise RuntimeError("reference checkpoint must be the dev-selected base_initial artifact")
    reference = freeze_reference(model.to(device))
    tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
    root = Path(config.data.manifest_dir)
    videos = sorted(
        (record for record in _load_records(root / "videos.jsonl", VideoRecord) if record.split == "train"),
        key=lambda record: record.official_order,
    )
    texts = sorted(
        (record for record in _load_records(root / "texts.jsonl", TextRecord) if record.split == "train"),
        key=lambda record: record.official_order,
    )
    if not videos or not texts:
        raise RuntimeError("reference caching requires non-empty train manifests")
    encoded_texts = []
    occurrences = []
    for text in texts:
        encoded = tokenize_with_spans(
            text.raw_text,
            text_uid=text.text_uid,
            tokenizer=tokenizer,
            max_positions=config.data.text_max_positions,
        )
        if config.data.auxiliary_full_caption_only and not encoded.fully_retained:
            continue
        encoded_texts.append((text, encoded))
        for span in encoded.lexical_spans:
            if span.fully_retained and span.token_positions:
                occurrences.append((text, encoded, span))
    batch_size = int(batch_size or config.evaluation.encode_batch_size)
    if batch_size < 1:
        raise ValueError("reference cache batch size must be positive")
    dimension = config.model.embedding_dim
    scratch = Path(tempfile.mkdtemp(prefix="method1-reference-build-"))
    try:
        video_tokens = np.lib.format.open_memmap(
            scratch / "videos.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(videos), config.data.feature_len, dimension),
        )
        video_valid = np.lib.format.open_memmap(
            scratch / "video_valid.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(len(videos), config.data.feature_len),
        )
        video_index: list[dict[str, Any]] = []
        for start in range(0, len(videos), batch_size):
            records = videos[start : start + batch_size]
            batch, selected = _video_batch(config, records)
            batch = {key: value.to(device) for key, value in batch.items()}
            encoded_video, valid = encode_reference_video(reference, batch)
            video_tokens[start : start + len(records)] = encoded_video.cpu().numpy()
            video_valid[start : start + len(records)] = valid.cpu().numpy()
            for offset, (record, indexes) in enumerate(zip(records, selected, strict=True)):
                video_index.append(
                    {
                        "row": start + offset,
                        "video_uid": record.video_uid,
                        "group_uid": record.group_uid,
                        "selected_feature_indices": indexes.tolist(),
                        "valid_length": int((indexes >= 0).sum()),
                        "agnostic_sha256": record.agnostic_sha256,
                        "aware_sha256": record.aware_sha256,
                    }
                )
        positive_spans = np.lib.format.open_memmap(
            scratch / "positive.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(occurrences), dimension),
        )
        positive_index: list[dict[str, Any]] = []
        occurrence_row = 0
        for start in range(0, len(encoded_texts), batch_size):
            current = encoded_texts[start : start + batch_size]
            ids = torch.tensor([item[1].input_ids for item in current], dtype=torch.long, device=device)
            valid = torch.tensor([item[1].text_valid for item in current], dtype=torch.bool, device=device)
            tokens, _ = encode_reference_text(reference, ids, torch.zeros_like(ids), valid)
            token_rows: list[torch.Tensor] = []
            span_positions: list[tuple[int, ...]] = []
            span_metadata = []
            for batch_row, (text, encoded) in enumerate(current):
                for span in encoded.lexical_spans:
                    if not span.fully_retained or not span.token_positions:
                        continue
                    token_rows.append(tokens[batch_row])
                    span_positions.append(span.token_positions)
                    span_metadata.append((text, encoded, span))
            if token_rows:
                pooled = pool_reference_spans(
                    reference, torch.stack(token_rows), span_positions
                ).cpu().numpy()
                positive_spans[occurrence_row : occurrence_row + len(pooled)] = pooled
                for local_row, (text, encoded, span) in enumerate(span_metadata):
                    positive_index.append(
                        {
                            "row": occurrence_row + local_row,
                            "occurrence_uid": span.occurrence_uid,
                            "text_uid": text.text_uid,
                            "group_uid": text.group_uid,
                            "caption_hash": encoded.caption_hash,
                            "word": span.surface,
                            "char_span": list(span.char_span),
                            "positions": list(span.token_positions),
                        }
                    )
                occurrence_row += len(pooled)
        if occurrence_row != len(occurrences):
            raise AssertionError("positive span counting and encoding diverged")
        video_tokens.flush()
        video_valid.flush()
        positive_spans.flush()
        metadata = write_reference_cache(
            config.reference.cache_dir,
            identity=identity,
            arrays={
                "reference_video_tokens": video_tokens,
                "reference_video_valid": video_valid,
                "reference_positive_spans": positive_spans,
            },
            video_index=video_index,
            positive_span_index=positive_index,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "schema_version": 1,
        "status": "complete",
        "cache_dir": str(Path(config.reference.cache_dir).resolve()),
        "identity_sha256": identity.digest,
        "teacher_checkpoint_sha256": identity.teacher_checkpoint_sha256,
        "video_count": len(videos),
        "eligible_caption_count": len(encoded_texts),
        "positive_occurrence_count": len(occurrences),
        "cache_content_sha256": metadata["content_sha256"],
        "cost": {
            "wall_seconds": time.perf_counter() - started_at,
            "video_encoder_batches": (len(videos) + batch_size - 1) // batch_size,
            "text_encoder_batches": (len(occurrences) + batch_size - 1) // batch_size,
            "artifact_bytes": sum(
                path.stat().st_size
                for path in Path(config.reference.cache_dir).rglob("*")
                if path.is_file()
            ),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(torch_device))
                if torch_device.type == "cuda"
                else 0
            ),
        },
    }
