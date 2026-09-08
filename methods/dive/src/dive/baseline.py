from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .adapters import (
    NativeTextFeatures,
    NativeVideoFeatures,
    SedsAdapter,
    SedsManifestInputBuilder,
    SedsTextBatch,
    SedsVideoBatch,
)
from .artifacts import ArtifactError, ArtifactResolver, ResolvedArtifact
from .config import config_hash
from .data.manifest import SampleRecord, load_manifest, manifest_hash
from .data.relevance import load_relevance, relevance_hash
from .data.temporal import CompactFrameMap, load_compact_frame_maps
from .eval.metrics import evaluate_retrieval


class BaselineValidationError(ValueError):
    """The controlled B0 validation stage cannot produce a trustworthy result."""


def _mapping(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise BaselineValidationError(f"{section} config must be a mapping")
    return value


def _configured_or_prepared(
    data: Mapping[str, Any],
    resolver: ArtifactResolver,
    field: str,
    output_name: str,
) -> Path:
    value = data.get(field)
    if isinstance(value, str) and value:
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        if not path.exists():
            raise BaselineValidationError(f"MISSING_RESOURCE: data.{field}: {path}")
        return path.resolve()
    try:
        return resolver.resolve("prepare_data", output_name, scope="shared").path
    except ArtifactError as exc:
        raise BaselineValidationError(
            f"MISSING_PARENT_ARTIFACT: data.{field} cannot resolve prepare_data.{output_name}"
        ) from exc


def _required_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BaselineValidationError(f"MISSING_RESOURCE: {field} is not configured")
    path = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if not path.exists():
        raise BaselineValidationError(f"MISSING_RESOURCE: {field}: {path}")
    return path


def _checkpoint_path(
    baseline: Mapping[str, Any], resolver: ArtifactResolver
) -> tuple[Path, ResolvedArtifact | None]:
    value = baseline.get("locked_checkpoint")
    if isinstance(value, str) and value:
        return _required_path(value, "baseline.locked_checkpoint"), None
    try:
        parent = resolver.resolve("baseline_train", "locked_checkpoint", scope="shared")
    except ArtifactError as exc:
        raise BaselineValidationError(
            "MISSING_PARENT_ARTIFACT: baseline.locked_checkpoint cannot resolve "
            "baseline_train.locked_checkpoint"
        ) from exc
    return parent.path, parent


def _video_to(batch: SedsVideoBatch, device: torch.device) -> SedsVideoBatch:
    return SedsVideoBatch(
        sample_ids=batch.sample_ids,
        right_pose=batch.right_pose.to(device),
        left_pose=batch.left_pose.to(device),
        body_pose=batch.body_pose.to(device),
        clip_starts=batch.clip_starts.to(device),
        legacy_video_mask=batch.legacy_video_mask.to(device),
        rgb_features=batch.rgb_features.to(device),
        grid_id=batch.grid_id,
        raw_frame_counts=batch.raw_frame_counts,
        frames_per_second=batch.frames_per_second,
        pose_raw_frame_indices=batch.pose_raw_frame_indices,
    )


def _text_to(batch: SedsTextBatch, device: torch.device) -> SedsTextBatch:
    return SedsTextBatch(
        text_ids=batch.text_ids,
        input_ids=batch.input_ids.to(device),
        token_type_ids=batch.token_type_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
    )


def _cpu_video(value: NativeVideoFeatures) -> NativeVideoFeatures:
    return NativeVideoFeatures(
        sample_ids=value.sample_ids,
        pooled=value.pooled.detach().cpu(),
        validity=value.validity.detach().cpu(),
        streams={name: tensor.detach().cpu() for name, tensor in value.streams.items()},
        metadata=dict(value.metadata),
    )


def _cpu_text(value: NativeTextFeatures) -> NativeTextFeatures:
    return NativeTextFeatures(
        text_ids=value.text_ids,
        pooled=value.pooled.detach().cpu(),
        token_features=value.token_features.detach().cpu(),
        token_validity=value.token_validity.detach().cpu(),
        metadata=dict(value.metadata),
    )


def _join_video(parts: Sequence[NativeVideoFeatures]) -> NativeVideoFeatures:
    if not parts:
        raise BaselineValidationError("no video features were encoded")
    stream_names = set(parts[0].streams)
    if any(set(part.streams) != stream_names for part in parts):
        raise BaselineValidationError("video feature batches have inconsistent streams")
    return NativeVideoFeatures(
        sample_ids=tuple(item for part in parts for item in part.sample_ids),
        pooled=torch.cat([part.pooled for part in parts]),
        validity=torch.cat([part.validity for part in parts]),
        streams={
            name: torch.cat([part.streams[name] for part in parts]) for name in stream_names
        },
        metadata=dict(parts[0].metadata),
    )


def _join_text(parts: Sequence[NativeTextFeatures]) -> NativeTextFeatures:
    if not parts:
        raise BaselineValidationError("no text features were encoded")
    return NativeTextFeatures(
        text_ids=tuple(item for part in parts for item in part.text_ids),
        pooled=torch.cat([part.pooled for part in parts]),
        token_features=torch.cat([part.token_features for part in parts]),
        token_validity=torch.cat([part.token_validity for part in parts]),
        metadata=dict(parts[0].metadata),
    )


def _slice_video(value: NativeVideoFeatures, start: int, end: int, device: torch.device) -> NativeVideoFeatures:
    return NativeVideoFeatures(
        sample_ids=value.sample_ids[start:end],
        pooled=value.pooled[start:end].to(device),
        validity=value.validity[start:end].to(device),
        streams={name: tensor[start:end].to(device) for name, tensor in value.streams.items()},
        metadata=value.metadata,
    )


def _slice_text(value: NativeTextFeatures, start: int, end: int, device: torch.device) -> NativeTextFeatures:
    return NativeTextFeatures(
        text_ids=value.text_ids[start:end],
        pooled=value.pooled[start:end].to(device),
        token_features=value.token_features[start:end].to(device),
        token_validity=value.token_validity[start:end].to(device),
        metadata=value.metadata,
    )


def score_feature_gallery(
    adapter: SedsAdapter,
    video: NativeVideoFeatures,
    text: NativeTextFeatures,
    *,
    device: torch.device,
    video_chunk: int,
    text_chunk: int,
) -> np.ndarray:
    """Compute the complete ID-addressed gallery without retaining pair tensors on GPU."""
    if video_chunk <= 0 or text_chunk <= 0:
        raise BaselineValidationError("gallery chunk sizes must be positive")
    scores = np.empty((len(video.sample_ids), len(text.text_ids)), dtype=np.float32)
    for video_start in range(0, len(video.sample_ids), video_chunk):
        video_end = min(video_start + video_chunk, len(video.sample_ids))
        video_part = _slice_video(video, video_start, video_end, device)
        for text_start in range(0, len(text.text_ids), text_chunk):
            text_end = min(text_start + text_chunk, len(text.text_ids))
            text_part = _slice_text(text, text_start, text_end, device)
            result = adapter.score_prelogit(video_part, text_part)
            block = result.scores.detach().float().cpu().numpy()
            if block.shape != (video_end - video_start, text_end - text_start):
                raise BaselineValidationError("baseline score block has an invalid shape")
            scores[video_start:video_end, text_start:text_end] = block
    if not np.isfinite(scores).all():
        raise BaselineValidationError("baseline full-gallery scores contain NaN/Inf")
    if float(scores.min()) < -1.00001 or float(scores.max()) > 1.00001:
        raise BaselineValidationError("prelogit baseline scores exceed cosine bounds")
    return scores


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_scores(path: Path, scores: np.ndarray, video_ids: Sequence[str], text_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            scores=scores,
            video_ids=np.asarray(video_ids, dtype=np.str_),
            text_ids=np.asarray(text_ids, dtype=np.str_),
        )
    os.replace(temporary, path)


def validate_seds_baseline(
    config: Mapping[str, Any],
    *,
    split: str = "dev",
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run controlled full-gallery B0 validation and register immutable scores/metrics."""
    if split != "dev":
        raise BaselineValidationError("controlled baseline selection validation is dev-only")
    if batch_size <= 0:
        raise BaselineValidationError("batch_size must be positive")
    data = _mapping(config, "data")
    baseline = _mapping(config, "baseline")
    evaluation = _mapping(config, "evaluation")
    if baseline.get("family") != "seds":
        raise BaselineValidationError("baseline validation currently requires family=seds")
    resolver = ArtifactResolver(config)
    try:
        data_audit = resolver.resolve("validate_data", "audit", scope="shared")
    except ArtifactError as exc:
        raise BaselineValidationError(
            "MISSING_PARENT_ARTIFACT: validate_data.audit must pass before baseline validation"
        ) from exc
    manifest_path = _configured_or_prepared(
        data, resolver, f"{split}_manifest", f"{split}_manifest"
    )
    relevance_dir = _configured_or_prepared(data, resolver, "relevance_dir", "relevance_dir")
    frame_maps_dir = _configured_or_prepared(
        data, resolver, "frame_maps_dir", "frame_maps_dir"
    )
    checkpoint, checkpoint_parent = _checkpoint_path(baseline, resolver)
    upstream_root = _required_path(data.get("upstream_root"), "data.upstream_root")
    reproduction = _required_path(
        baseline.get("reproduction_config"), "baseline.reproduction_config"
    )
    pose_root = _required_path(data.get("pose_root"), "data.pose_root")
    rgb_root = _required_path(data.get("rgb_cache_root"), "data.rgb_cache_root")
    records = load_manifest(manifest_path, expected_split=split)
    frame_maps = load_compact_frame_maps(
        frame_maps_dir / f"{split}.jsonl",
        expected_sample_ids=[record.sample_id for record in records],
    )
    map_by_id: dict[str, CompactFrameMap] = {item.sample_id: item for item in frame_maps}
    relevance_path = relevance_dir / f"{split}.jsonl"
    text_records = tuple(dict.fromkeys(record.text_id for record in records))
    first_by_text: dict[str, SampleRecord] = {}
    for record in records:
        existing = first_by_text.setdefault(record.text_id, record)
        if existing.text_model != record.text_model:
            raise BaselineValidationError(
                f"text ID maps to inconsistent model text: {record.text_id}"
            )
    unique_text_records = tuple(first_by_text[text_id] for text_id in text_records)
    relevance = load_relevance(
        relevance_path,
        video_ids=[record.video_id for record in records],
        text_ids=list(text_records),
    )
    video_ids = tuple(record.video_id for record in records)
    target_device = torch.device(device)
    adapter = SedsAdapter.from_official_checkpoint(
        checkpoint,
        config,
        upstream_root=upstream_root,
        device=target_device,
    )
    builder = SedsManifestInputBuilder(
        upstream_root=upstream_root,
        reproduction_config=reproduction,
        pose_root=pose_root,
        rgb_root=rgb_root,
    )

    video_parts: list[NativeVideoFeatures] = []
    for start in range(0, len(records), batch_size):
        part = records[start : start + batch_size]
        native = builder.build_video_batch(
            part,
            raw_frame_counts=[map_by_id[item.sample_id].pose_input_step_count for item in part],
            frames_per_second=[map_by_id[item.sample_id].fps for item in part],
        )
        video_parts.append(_cpu_video(adapter.encode_video_native(_video_to(native, target_device))))
    text_parts: list[NativeTextFeatures] = []
    for start in range(0, len(unique_text_records), batch_size):
        part = unique_text_records[start : start + batch_size]
        native = builder.build_text_batch(part)
        text_parts.append(_cpu_text(adapter.encode_text_native(_text_to(native, target_device))))
    video_features = _join_video(video_parts)
    text_features = _join_text(text_parts)
    probe_video = _slice_video(
        video_features, 0, min(4, len(video_features.sample_ids)), target_device
    )
    probe_text = _slice_text(
        text_features, 0, min(4, len(text_features.text_ids)), target_device
    )
    parity = adapter.validate_unpadded_native_parity(probe_video, probe_text)
    scores = score_feature_gallery(
        adapter,
        video_features,
        text_features,
        device=target_device,
        video_chunk=int(evaluation.get("query_chunk", 16)),
        text_chunk=int(evaluation.get("candidate_chunk", 128)),
    )
    metrics = evaluate_retrieval(
        scores,
        video_ids,
        text_features.text_ids,
        relevance,
        topk=evaluation.get("topk", (1, 5, 10)),
    )
    output_dir = resolver.output_path("shared", "baseline", "dev")
    score_path = output_dir / "prelogit_scores.npz"
    report_path = output_dir / "validation.json"
    _atomic_scores(score_path, scores, video_ids, text_features.text_ids)
    report: dict[str, Any] = {
        "schema_version": "seds_baseline_validation.v1",
        "ready": True,
        "config_sha256": config_hash(config),
        "split": split,
        "selection_eligible": True,
        "test_content_used_for_tuning": False,
        "checkpoint": dict(adapter.checkpoint_metadata),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash(manifest_path),
            "video_count": len(video_features.sample_ids),
            "text_count": len(text_features.text_ids),
        },
        "relevance": {"path": str(relevance_path), "sha256": relevance_hash(relevance_path)},
        "native_unpadded_parity": parity,
        "score": {
            "path": str(score_path),
            "shape": list(scores.shape),
            "orientation": "video_rows_text_columns",
            "scale": "prelogit",
            "minimum": float(scores.min()),
            "median": float(np.median(scores)),
            "maximum": float(scores.max()),
        },
        "metrics": metrics.to_dict(),
    }
    _atomic_json(report_path, report)
    parents = [data_audit]
    if checkpoint_parent is not None:
        parents.append(checkpoint_parent)
    registered = resolver.record_stage(
        "baseline_validate",
        {"report": report_path, "prelogit_scores": score_path},
        scope="shared",
        parents=parents,
        metadata={"split": split, "video_count": len(records), "text_count": len(text_records)},
    )
    report["artifacts"] = {
        name: {"artifact_id": record.artifact_id, "sha256": record.sha256}
        for name, record in registered.items()
    }
    return report
