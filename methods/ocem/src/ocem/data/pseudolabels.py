"""Leakage-safe P14T pseudo-label indexing from audited I3D embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ocem.data.features import FeatureAuditError, load_feature_payload
from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class PseudoLabelError(ValueError):
    """Raised when pseudo-label inputs violate the locked P14T protocol."""


LOCKED_RECIPE = {
    "confidence_threshold": 0.6,
    "clip_frames": 16,
    "window_stride": 1,
    "merge_start_distance_frames": 3,
    "suppress_start_distance_frames": 24,
    "selection": "per-class/per-video descending confidence",
}


def passes_confidence_threshold(confidence: float, threshold: float = 0.6) -> bool:
    """Match CiCo's strict ``top_value > threshold`` acceptance predicate."""

    return math.isfinite(confidence) and confidence > threshold


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PseudoLabelError(f"invalid JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise PseudoLabelError(f"JSON document must be an object: {path}")
    return payload


def _manifest_records(path: Path) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PseudoLabelError(f"cannot read manifest {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PseudoLabelError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(record, Mapping) or not record.get("sample_id"):
            raise PseudoLabelError(f"missing sample_id at {path}:{line_number}")
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise PseudoLabelError(f"duplicate sample_id {sample_id!r} in {path}")
        records[sample_id] = record
    if not records:
        raise PseudoLabelError(f"manifest is empty: {path}")
    return records


def _class_names(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PseudoLabelError(f"cannot read class vocabulary {path}: {error}") from error
    names: list[str] = []
    for index, line in enumerate(lines):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise PseudoLabelError(f"malformed class vocabulary at line {index + 1}")
        try:
            label = int(fields[0])
        except ValueError as error:
            raise PseudoLabelError(f"non-integer class label at line {index + 1}") from error
        if label != index:
            raise PseudoLabelError(f"class labels must be contiguous at line {index + 1}")
        names.append(fields[1])
    if not names or len(set(names)) != len(names):
        raise PseudoLabelError("class vocabulary must be non-empty and unique")
    return names


def _validated_plan(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PseudoLabelError(f"adaptation plan SHA-256 mismatch: {actual_sha256}")
    plan = _read_json(path)
    required = {
        "schema_version": "ocem.p14t_adaptation_plan.v1",
        "status": "PASS",
        "dataset": "phoenix2014t",
        "target_data_policy": "train_manifest_only",
        "validation_or_test_used": False,
        "ready_for_pseudo_label_generation": True,
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise PseudoLabelError(f"adaptation plan has invalid {key}: {plan.get(key)!r}")
    recipe = plan.get("pseudo_label_recipe")
    if not isinstance(recipe, Mapping):
        raise PseudoLabelError("adaptation plan lacks pseudo-label recipe")
    for key, expected in LOCKED_RECIPE.items():
        if recipe.get(key) != expected:
            raise PseudoLabelError(f"pseudo-label recipe drift for {key}: {recipe.get(key)!r}")

    source = plan.get("source_manifest")
    if not isinstance(source, Mapping):
        raise PseudoLabelError("adaptation plan lacks source manifest provenance")
    source_path = Path(str(source.get("path", "")))
    if sha256_file(source_path) != source.get("sha256"):
        raise PseudoLabelError("source train manifest drifted after adaptation planning")
    source_records = _manifest_records(source_path)
    if len(source_records) != source.get("sample_count"):
        raise PseudoLabelError("source train manifest count differs from adaptation plan")
    if canonical_json_sha256(sorted(source_records)) != source.get("sample_ids_sha256"):
        raise PseudoLabelError("source train sample IDs differ from adaptation plan")

    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise PseudoLabelError("adaptation plan has no sample assignments")
    assignment_map: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, Mapping) or item.get("adaptation_split") not in {
            "train",
            "holdout",
        }:
            raise PseudoLabelError("malformed adaptation assignment")
        sample_id = str(item.get("sample_id", ""))
        if not sample_id or sample_id in assignment_map:
            raise PseudoLabelError(f"missing or duplicate adaptation sample ID: {sample_id!r}")
        assignment_map[sample_id] = str(item["adaptation_split"])
    if set(assignment_map) != set(source_records):
        raise PseudoLabelError("adaptation assignments do not equal source train IDs")
    split_policy = plan.get("split_policy")
    if not isinstance(split_policy, Mapping):
        raise PseudoLabelError("adaptation plan lacks split policy")
    if canonical_json_sha256(assignments) != split_policy.get("assignment_sha256"):
        raise PseudoLabelError("adaptation assignment digest mismatch")
    counts = Counter(assignment_map.values())
    if counts["train"] != split_policy.get("train_count") or counts["holdout"] != split_policy.get(
        "holdout_count"
    ):
        raise PseudoLabelError("adaptation assignment counts mismatch")

    forbidden_ids: set[str] = set()
    forbidden = plan.get("forbidden_manifests")
    if not isinstance(forbidden, list) or not forbidden:
        raise PseudoLabelError("adaptation plan lacks forbidden validation/test manifests")
    for item in forbidden:
        if not isinstance(item, Mapping):
            raise PseudoLabelError("malformed forbidden manifest provenance")
        forbidden_path = Path(str(item.get("path", "")))
        if sha256_file(forbidden_path) != item.get("sha256"):
            raise PseudoLabelError(f"forbidden manifest drifted: {forbidden_path}")
        records = _manifest_records(forbidden_path)
        if len(records) != item.get("sample_count"):
            raise PseudoLabelError(f"forbidden manifest count drifted: {forbidden_path}")
        forbidden_ids.update(records)
    overlap = sorted(set(assignment_map) & forbidden_ids)
    if overlap:
        raise PseudoLabelError(f"adaptation plan leaks forbidden IDs: {overlap[:10]}")
    return plan, {
        "records": source_records,
        "assignments": assignment_map,
        "forbidden_ids": forbidden_ids,
        "sha256": actual_sha256,
    }


def select_cico_segments(
    windows: Sequence[Mapping[str, Any]],
    *,
    merge_distance: int = 3,
    suppress_distance: int = 24,
) -> list[dict[str, Any]]:
    """Port the observed CiCo per-class/video temporal grouping exactly.

    The upstream implementation appends every <=3-frame merge to the first,
    highest-confidence anchor even when proximity was detected against a later
    anchor. This non-local behavior is intentionally retained and surfaced in
    the output rather than silently corrected.
    """

    if merge_distance < 0 or suppress_distance < merge_distance:
        raise PseudoLabelError("invalid temporal grouping distances")
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: (-float(item["confidence"]), int(item["start"])))
    for item in ordered:
        start, end = int(item["start"]), int(item["end"])
        if start < 0 or end <= start:
            raise PseudoLabelError(f"invalid pseudo-label support [{start}, {end})")
    first = ordered[0]
    first_anchor = int(first["start"])
    groups: dict[int, dict[str, Any]] = {
        first_anchor: {
            "anchor_start": first_anchor,
            "confidence": float(first["confidence"]),
            "window_starts": [first_anchor],
            "window_ends": [int(first["end"])],
        }
    }
    for item in ordered[1:]:
        start, end = int(item["start"]), int(item["end"])
        selected = False
        for anchor in tuple(groups):
            distance = abs(start - anchor)
            if distance <= merge_distance:
                # Exact epoch_pseudo.py behavior: merge into `frame_start`, the
                # initial anchor, rather than the proximate `frame_start_key`.
                groups[first_anchor]["window_starts"].append(start)
                groups[first_anchor]["window_ends"].append(end)
                selected = True
            elif distance <= suppress_distance:
                selected = True
        if not selected:
            groups[start] = {
                "anchor_start": start,
                "confidence": float(item["confidence"]),
                "window_starts": [start],
                "window_ends": [end],
            }

    segments: list[dict[str, Any]] = []
    for group in groups.values():
        support_start = min(group["window_starts"])
        support_end = max(group["window_ends"])
        segments.append(
            {
                "anchor_start": group["anchor_start"],
                "confidence": group["confidence"],
                "support_start_frame": support_start,
                "support_end_frame_exclusive": support_end,
                "upstream_clip_start_frame": support_start,
                # epoch_pseudo passes max(frame_indices) as Python's exclusive
                # slice end, so a lone 16-frame window materializes 15 frames.
                "upstream_clip_end_frame_exclusive": support_end - 1,
                "merged_window_count": len(group["window_starts"]),
                "merged_window_starts": group["window_starts"],
            }
        )
    return segments


def _load_classifier(checkpoint: Path, expected_sha256: str, classes: int, device: str):
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_sha256:
        raise PseudoLabelError(f"checkpoint SHA-256 mismatch: {checkpoint_sha256}")
    try:
        import torch
    except ImportError as error:
        raise PseudoLabelError("PyTorch is required for pseudo-label generation") from error
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise PseudoLabelError("I3D checkpoint must contain a state_dict")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload["state_dict"].items()
    }
    weight = state.get("logits.conv3d.weight")
    bias = state.get("logits.conv3d.bias")
    if weight is None or tuple(weight.shape) != (classes, 1024, 1, 1, 1):
        raise PseudoLabelError(f"classifier weight must have shape {(classes, 1024, 1, 1, 1)}")
    if bias is None or tuple(bias.shape) != (classes,):
        raise PseudoLabelError(f"classifier bias must have shape {(classes,)}")
    try:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise PseudoLabelError(f"CUDA device requested but unavailable: {device}")
        weight = weight.flatten(1).float().to(resolved_device)
        bias = bias.float().to(resolved_device)
    except (RuntimeError, ValueError) as error:
        raise PseudoLabelError(f"cannot initialize classifier on {device}: {error}") from error
    return torch, weight, bias, resolved_device, checkpoint_sha256


def _inventory(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise PseudoLabelError(f"artifact directory is missing: {directory}")
    return {path.name[: -len(suffix)] for path in directory.glob(f"*{suffix}")}


def _load_sample(
    sample_id: str,
    manifest_record: Mapping[str, Any],
    feature_dir: Path,
    temporal_dir: Path,
    checkpoint_sha256: str,
) -> tuple[np.ndarray, list[int], list[int], str]:
    feature_path = feature_dir / f"{sample_id}.pkl"
    metadata_path = Path(f"{feature_path}.meta.json")
    support_path = temporal_dir / f"{sample_id}.json"
    metadata = _read_json(metadata_path)
    support = _read_json(support_path)
    if metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise PseudoLabelError(f"checkpoint provenance mismatch for {sample_id}")
    if metadata.get("stream_name") != "domain_agnostic":
        raise PseudoLabelError(f"non-agnostic feature stream for {sample_id}")
    if sha256_file(feature_path) != metadata.get("feature_sha256"):
        raise PseudoLabelError(f"feature bytes drifted after audit for {sample_id}")
    try:
        payload = load_feature_payload(feature_path)
    except (OSError, FeatureAuditError, ValueError) as error:
        raise PseudoLabelError(f"cannot load feature payload for {sample_id}: {error}") from error
    feature = payload.get("feature")
    starts, ends = support.get("rf_start"), support.get("rf_end")
    if (
        not isinstance(feature, np.ndarray)
        or feature.dtype != np.float32
        or feature.ndim != 2
        or feature.shape[1:] != (1024,)
        or not bool(np.isfinite(feature).all())
    ):
        raise PseudoLabelError(f"invalid [M,1024] float32 feature array for {sample_id}")
    if not isinstance(starts, list) or not isinstance(ends, list) or len(starts) != len(ends):
        raise PseudoLabelError(f"invalid temporal supports for {sample_id}")
    if feature.shape[0] != len(starts):
        raise PseudoLabelError(f"feature/support count mismatch for {sample_id}")
    if (
        support.get("interval_convention") != "half_open"
        or support.get("coordinate_system") != "input_frame"
    ):
        raise PseudoLabelError(f"unsupported temporal coordinate contract for {sample_id}")
    frame_count = support.get("decoded_frame_count")
    if not isinstance(frame_count, int) or frame_count < 1:
        raise PseudoLabelError(f"invalid decoded frame count for {sample_id}")
    expected_starts = list(range(max(frame_count - 15, 1)))
    expected_ends = [min(start + 16, frame_count) for start in expected_starts]
    if starts != expected_starts or ends != expected_ends:
        raise PseudoLabelError(f"support windows drifted for {sample_id}")
    source_video = str(support.get("source_video", ""))
    if source_video != metadata.get("source_video") or Path(str(payload.get("name", ""))) != Path(
        source_video
    ):
        raise PseudoLabelError(f"source video provenance mismatch for {sample_id}")
    raw_relpath = Path(str(manifest_record.get("raw_relpath", "")))
    source_path = Path(source_video)
    if (
        not raw_relpath.parts
        or tuple(source_path.parts[-len(raw_relpath.parts) :]) != raw_relpath.parts
    ):
        raise PseudoLabelError(f"source video is not the train manifest path for {sample_id}")
    return feature, starts, ends, source_video


def _atomic_jsonl(records: Iterable[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def generate_p14t_pseudolabel_index(
    *,
    adaptation_plan: str | Path,
    expected_plan_sha256: str,
    feature_dir: str | Path,
    temporal_dir: str | Path,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    output_index: str | Path,
    device: str = "cuda:0",
    batch_windows: int = 8192,
    report_interval: int = 250,
) -> dict[str, Any]:
    """Generate an immutable pseudo-segment index without touching dev/test features."""

    if batch_windows < 1 or report_interval < 1:
        raise PseudoLabelError("batch_windows and report_interval must be positive")
    plan_path = Path(adaptation_plan)
    feature_dir, temporal_dir = Path(feature_dir), Path(temporal_dir)
    checkpoint, output_index = Path(checkpoint), Path(output_index)
    plan, resolved = _validated_plan(plan_path, expected_plan_sha256)
    vocabulary = plan.get("class_vocabulary")
    checkpoint_plan = plan.get("checkpoint")
    if not isinstance(vocabulary, Mapping) or not isinstance(checkpoint_plan, Mapping):
        raise PseudoLabelError("adaptation plan lacks checkpoint or vocabulary provenance")
    vocabulary_path = Path(str(vocabulary.get("path", "")))
    if sha256_file(vocabulary_path) != vocabulary.get("sha256"):
        raise PseudoLabelError("class vocabulary drifted after adaptation planning")
    names = _class_names(vocabulary_path)
    if len(names) != vocabulary.get("classes") or canonical_json_sha256(names) != vocabulary.get(
        "names_sha256"
    ):
        raise PseudoLabelError("class vocabulary content differs from adaptation plan")
    if expected_checkpoint_sha256 != checkpoint_plan.get("sha256"):
        raise PseudoLabelError("requested checkpoint differs from adaptation plan")
    if checkpoint.resolve() != Path(str(checkpoint_plan.get("path", ""))).resolve():
        raise PseudoLabelError("requested checkpoint path differs from adaptation plan")

    source_ids = set(resolved["records"])
    feature_ids = _inventory(feature_dir, ".pkl")
    metadata_ids = _inventory(feature_dir, ".pkl.meta.json")
    support_ids = _inventory(temporal_dir, ".json")
    for artifact, ids in (
        ("features", feature_ids),
        ("feature metadata", metadata_ids),
        ("temporal supports", support_ids),
    ):
        if ids != source_ids:
            raise PseudoLabelError(
                f"{artifact} IDs do not exactly equal train manifest IDs "
                f"(missing={len(source_ids - ids)}, extra={len(ids - source_ids)})"
            )

    torch, weight, bias, resolved_device, checkpoint_sha256 = _load_classifier(
        checkpoint, expected_checkpoint_sha256, len(names), device
    )
    torch.set_float32_matmul_precision("highest")
    threshold = float(LOCKED_RECIPE["confidence_threshold"])
    started = time.monotonic()
    segments: list[dict[str, Any]] = []
    accepted_windows = 0
    total_windows = 0
    closest_threshold_distance = math.inf
    class_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    sample_ids = sorted(source_ids)

    with torch.inference_mode():
        for sample_number, sample_id in enumerate(sample_ids, start=1):
            feature, starts, ends, source_video = _load_sample(
                sample_id,
                resolved["records"][sample_id],
                feature_dir,
                temporal_dir,
                checkpoint_sha256,
            )
            predictions: list[dict[str, Any]] = []
            for offset in range(0, len(feature), batch_windows):
                matrix = torch.from_numpy(feature[offset : offset + batch_windows]).to(
                    resolved_device
                )
                logits = matrix @ weight.T + bias
                maximum, labels = logits.max(dim=1)
                confidence = torch.exp(maximum - torch.logsumexp(logits, dim=1))
                values = confidence.float().cpu().numpy()
                label_values = labels.cpu().numpy()
                if len(values):
                    closest_threshold_distance = min(
                        closest_threshold_distance,
                        float(np.min(np.abs(values.astype(np.float64) - threshold))),
                    )
                for local, (value, label) in enumerate(zip(values, label_values, strict=True)):
                    index = offset + local
                    if passes_confidence_threshold(float(value), threshold):
                        predictions.append(
                            {
                                "confidence": float(value),
                                "class_index": int(label),
                                "start": int(starts[index]),
                                "end": int(ends[index]),
                            }
                        )
            total_windows += len(feature)
            accepted_windows += len(predictions)
            by_class: dict[int, list[dict[str, Any]]] = {}
            for prediction in predictions:
                by_class.setdefault(prediction["class_index"], []).append(prediction)
            for class_index in sorted(by_class):
                selected = select_cico_segments(by_class[class_index])
                for segment in selected:
                    identity = {
                        "source_sample_id": sample_id,
                        "class_index": class_index,
                        "anchor_start": segment["anchor_start"],
                    }
                    pseudo_id = hashlib.sha256(
                        ("ocem-p14t-pseudo-v1\0" + canonical_json_sha256(identity)).encode()
                    ).hexdigest()
                    record = {
                        "schema_version": "ocem.p14t_pseudolabel.v1",
                        "pseudo_id": pseudo_id,
                        "source_sample_id": sample_id,
                        "source_video": source_video,
                        "adaptation_split": resolved["assignments"][sample_id],
                        "class_index": class_index,
                        "class_name": names[class_index],
                        "selection_confidence": segment["confidence"],
                        "anchor_start_frame": segment["anchor_start"],
                        "support_start_frame": segment["support_start_frame"],
                        "support_end_frame_exclusive": segment["support_end_frame_exclusive"],
                        "upstream_clip_start_frame": segment["upstream_clip_start_frame"],
                        "upstream_clip_end_frame_exclusive": segment[
                            "upstream_clip_end_frame_exclusive"
                        ],
                        "merged_window_count": segment["merged_window_count"],
                        "merged_window_starts": segment["merged_window_starts"],
                        "materialization_semantics": "cico_epoch_pseudo_python_slice_v1",
                    }
                    segments.append(record)
                    class_counts[class_index] += 1
                    split_counts[record["adaptation_split"]] += 1
            if sample_number % report_interval == 0 or sample_number == len(sample_ids):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"pseudo-labels {sample_number}/{len(sample_ids)} samples, "
                    f"{total_windows} windows, {len(segments)} segments, "
                    f"{total_windows / elapsed:.1f} windows/s",
                    flush=True,
                )

    segments.sort(
        key=lambda item: (
            item["source_sample_id"],
            item["class_index"],
            item["anchor_start_frame"],
        )
    )
    if len({record["pseudo_id"] for record in segments}) != len(segments):
        raise PseudoLabelError("pseudo-label identity collision")
    if set(record["source_sample_id"] for record in segments) & resolved["forbidden_ids"]:
        raise PseudoLabelError("generated pseudo-labels contain validation/test IDs")
    _atomic_jsonl(segments, output_index)
    index_sha256 = sha256_file(output_index)
    elapsed = time.monotonic() - started
    return {
        "schema_version": "ocem.p14t_pseudolabel_index_report.v1",
        "status": "PASS",
        "dataset": "phoenix2014t",
        "source_policy": "audited_domain_agnostic_train_features_only",
        "adaptation_plan": {"path": str(plan_path.resolve()), "sha256": resolved["sha256"]},
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "classifier_shape": [len(names), 1024],
            "classifier_application": "embedding @ weight.T + bias",
        },
        "recipe": {
            **LOCKED_RECIPE,
            "compatibility": "exact observed epoch_pseudo.py grouping",
            "upstream_merge_target": "first_highest_confidence_anchor",
            "upstream_materialized_right_boundary": "max_frame_index_used_as_exclusive_end",
        },
        "input": {
            "feature_dir": str(feature_dir.resolve()),
            "temporal_dir": str(temporal_dir.resolve()),
            "samples": len(sample_ids),
            "windows": total_windows,
            "sample_ids_sha256": canonical_json_sha256(sample_ids),
        },
        "output": {
            "index_path": str(output_index.resolve()),
            "index_sha256": index_sha256,
            "segments": len(segments),
            "accepted_windows": accepted_windows,
            "samples_with_segments": len({item["source_sample_id"] for item in segments}),
            "classes_with_segments": len(class_counts),
            "adaptation_split_segments": dict(sorted(split_counts.items())),
            "class_segment_counts_sha256": canonical_json_sha256(sorted(class_counts.items())),
            "records_sha256": canonical_json_sha256(segments),
        },
        "numerics": {
            "dtype": "float32",
            "device": str(resolved_device),
            "batch_windows": batch_windows,
            "float32_matmul_precision": "highest",
            "closest_confidence_distance_to_threshold": closest_threshold_distance,
        },
        "provenance_checks": {
            "feature_bytes_rehashed": True,
            "feature_supports_revalidated": True,
            "feature_ids_equal_train_manifest": True,
            "validation_test_overlap_count": 0,
            "validation_or_test_features_read": False,
        },
        "elapsed_seconds": elapsed,
        "windows_per_second": total_windows / max(elapsed, 1e-9),
        "ready_for_loader_smoke": bool(segments),
        "ready_for_adaptation_training": False,
        "training_blocker": (
            "Manifest-indexed raw-frame pseudo-clip loader and one-batch "
            "loss/gradient/update parity have not passed yet."
        ),
    }
