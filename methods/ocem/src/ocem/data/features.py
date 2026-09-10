"""Audit deterministic I3D feature caches against locked sample manifests."""

from __future__ import annotations

import io
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class FeatureAuditError(ValueError):
    """Raised when an audit request itself is malformed."""


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Load the released NumPy cache format without arbitrary globals."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"disallowed pickle global: {module}.{name}")
        return super().find_class(module, name)


def load_feature_payload(path: str | Path) -> Mapping[str, Any]:
    """Load the NumPy-only cache payload without permitting arbitrary pickle globals."""

    path = Path(path)
    payload = _NumpyOnlyUnpickler(io.BytesIO(path.read_bytes())).load()
    if not isinstance(payload, Mapping):
        raise FeatureAuditError("feature pickle must contain a mapping")
    return payload


# Kept as an internal alias for callers from the original cache-audit module.
_safe_load_feature = load_feature_payload


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureAuditError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise FeatureAuditError(f"JSON document must be an object: {path}")
    return value


def _read_manifest(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FeatureAuditError(f"cannot read manifest {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise FeatureAuditError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(record, Mapping) or not record.get("sample_id"):
            raise FeatureAuditError(f"manifest record lacks sample_id at {path}:{line_number}")
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise FeatureAuditError(f"duplicate sample_id {sample_id!r} in {path}")
        seen.add(sample_id)
        records.append(record)
    if not records:
        raise FeatureAuditError(f"manifest is empty: {path}")
    return records


def _inventory_ids(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FeatureAuditError(f"cache directory is missing: {directory}")
    paths: dict[str, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        sample_id = path.name[: -len(suffix)]
        if sample_id in paths:
            raise FeatureAuditError(f"duplicate cache ID {sample_id!r} in {directory}")
        paths[sample_id] = path
    return paths


def _audit_one(
    record: Mapping[str, Any],
    feature_path: Path,
    metadata_path: Path,
    support_path: Path,
    expected_checkpoint_sha256: str,
    expected_stream_name: str,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    metadata = _read_json(metadata_path)
    support = _read_json(support_path)
    feature_sha256 = sha256_file(feature_path)
    errors: list[str] = []

    if metadata.get("feature_sha256") != feature_sha256:
        errors.append("feature_sha256_mismatch")
    if metadata.get("schema_version") != 1 or support.get("schema_version") != 1:
        errors.append("metadata_schema_version_mismatch")
    if metadata.get("checkpoint_sha256") != expected_checkpoint_sha256:
        errors.append("checkpoint_sha256_mismatch")
    if metadata.get("feature_dtype") != "float32":
        errors.append("feature_dtype_not_float32")
    if metadata.get("stream_name") != expected_stream_name:
        errors.append("unexpected_stream_name")
    if metadata.get("recipe_sha256") != support.get("recipe_sha256"):
        errors.append("recipe_sha256_mismatch")
    if metadata.get("source_video_sha256") != support.get("source_video_sha256"):
        errors.append("source_video_sha256_mismatch")
    if metadata.get("source_video") != support.get("source_video"):
        errors.append("source_video_path_mismatch")

    raw_relpath = Path(str(record.get("raw_relpath", "")))
    source_video = Path(str(metadata.get("source_video", "")))
    raw_parts = raw_relpath.parts
    if not raw_parts or tuple(source_video.parts[-len(raw_parts) :]) != raw_parts:
        errors.append("source_video_not_manifest_path")
    elif not source_video.is_file():
        errors.append("source_video_missing")
    elif record.get("raw_bytes") is not None and source_video.stat().st_size != record["raw_bytes"]:
        errors.append("source_video_size_mismatch")

    if support.get("verified") is not True:
        errors.append("support_not_verified")
    if support.get("coordinate_system") != "input_frame":
        errors.append("support_coordinate_system_mismatch")
    if support.get("interval_convention") != "half_open":
        errors.append("support_interval_convention_mismatch")
    recipe = support.get("recipe")
    if not isinstance(recipe, Mapping):
        errors.append("support_recipe_missing")
        recipe = {}
    elif canonical_json_sha256(recipe) != support.get("recipe_sha256"):
        errors.append("support_recipe_hash_mismatch")
    if recipe.get("clip_frames") != 16 or recipe.get("stride") != 1:
        errors.append("support_window_recipe_mismatch")

    starts = support.get("rf_start")
    ends = support.get("rf_end")
    frame_count = support.get("decoded_frame_count")
    if not isinstance(starts, list) or not isinstance(ends, list) or len(starts) != len(ends):
        errors.append("support_arrays_invalid")
        starts, ends = [], []
    if not isinstance(frame_count, int) or frame_count <= 0:
        errors.append("decoded_frame_count_invalid")
        frame_count = 0
    if record.get("frame_count") != frame_count:
        errors.append("decoded_frame_count_manifest_mismatch")
    fps = support.get("fps")
    expected_fps = record.get("fps_num", 0) / max(record.get("fps_den", 0), 1)
    if not isinstance(fps, (int, float)) or not np.isclose(fps, expected_fps, atol=1e-9, rtol=0):
        errors.append("fps_manifest_mismatch")

    expected_windows = max(frame_count - 15, 1) if frame_count else 0
    expected_starts = list(range(expected_windows))
    expected_ends = [min(start + 16, frame_count) for start in expected_starts]
    if starts != expected_starts or ends != expected_ends:
        errors.append("support_intervals_not_stride1_actual_frames")

    try:
        payload = _safe_load_feature(feature_path)
    except (OSError, pickle.UnpicklingError, FeatureAuditError, ValueError) as error:
        errors.append(f"feature_payload_invalid:{type(error).__name__}")
        feature = None
    else:
        feature = payload.get("feature")
        payload_name = Path(str(payload.get("name", "")))
        if payload_name != source_video:
            errors.append("feature_payload_source_mismatch")
    shape = metadata.get("feature_shape")
    if not isinstance(feature, np.ndarray):
        errors.append("feature_array_missing")
        actual_shape: list[int] = []
    else:
        actual_shape = list(feature.shape)
        if feature.dtype != np.float32:
            errors.append("feature_array_dtype_mismatch")
        if feature.ndim != 2 or feature.shape[1:] != (1024,):
            errors.append("feature_array_shape_contract_mismatch")
        if feature.shape[0] != len(starts):
            errors.append("feature_support_count_mismatch")
        if not bool(np.isfinite(feature).all()):
            errors.append("feature_array_nonfinite")
    if shape != actual_shape:
        errors.append("feature_shape_metadata_mismatch")

    return {
        "sample_id": sample_id,
        "status": "PASS" if not errors else "FAIL_TECHNICAL",
        "errors": errors,
        "feature_sha256": feature_sha256,
        "metadata_sha256": sha256_file(metadata_path),
        "support_sha256": sha256_file(support_path),
        "source_video_sha256": metadata.get("source_video_sha256"),
        "recipe_sha256": metadata.get("recipe_sha256"),
        "shape": actual_shape,
        "windows": len(starts),
        "feature_bytes": feature_path.stat().st_size,
    }


def audit_feature_cache(
    *,
    manifest_dir: str | Path,
    feature_root: str | Path,
    temporal_root: str | Path,
    expected_checkpoint_sha256: str,
    split_dirs: Mapping[str, str],
    workers: int = 8,
    expected_stream_name: str = "domain_agnostic",
    adaptation_report: str | Path | None = None,
    expected_adaptation_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify an existing cache without accepting it as target-adapted features."""

    if len(expected_checkpoint_sha256) != 64:
        raise FeatureAuditError("expected checkpoint SHA-256 must contain 64 hex digits")
    try:
        int(expected_checkpoint_sha256, 16)
    except ValueError as error:
        raise FeatureAuditError("expected checkpoint SHA-256 must be hexadecimal") from error
    if workers <= 0:
        raise FeatureAuditError("workers must be positive")
    if expected_stream_name not in {"domain_agnostic", "domain_adapted_p14t"}:
        raise FeatureAuditError(f"unsupported feature stream: {expected_stream_name}")
    adaptation_provenance: dict[str, Any] | None = None
    if expected_stream_name == "domain_adapted_p14t":
        if adaptation_report is None or expected_adaptation_report_sha256 is None:
            raise FeatureAuditError("adapted stream requires a hashed adaptation report")
        adaptation_path = Path(adaptation_report)
        actual_report_sha256 = sha256_file(adaptation_path)
        if actual_report_sha256 != expected_adaptation_report_sha256:
            raise FeatureAuditError("adaptation report SHA-256 mismatch")
        report = _read_json(adaptation_path)
        final_checkpoint = report.get("final_checkpoint")
        data_policy = report.get("data_policy")
        config = report.get("config")
        if (
            report.get("schema_version") != "ocem.p14t_i3d_adaptation_run.v1"
            or report.get("status") != "PASS"
            or report.get("completed_epochs") != 15
            or not isinstance(final_checkpoint, Mapping)
            or final_checkpoint.get("sha256") != expected_checkpoint_sha256
            or final_checkpoint.get("fixed_epoch") != 15
            or not isinstance(data_policy, Mapping)
            or data_policy.get("holdout_used_for_optimizer") is not False
            or data_policy.get("holdout_used_for_checkpoint_selection") is not False
            or data_policy.get("validation_or_test_used") is not False
            or not isinstance(config, Mapping)
            or config.get("checkpoint_selection") != "fixed_final_epoch_15"
        ):
            raise FeatureAuditError(
                "adaptation report does not prove fixed train-only epoch-15 provenance"
            )
        adaptation_provenance = {
            "path": str(adaptation_path.resolve()),
            "sha256": actual_report_sha256,
            "fixed_epoch": 15,
            "validation_or_test_used": False,
        }
    elif adaptation_report is not None or expected_adaptation_report_sha256 is not None:
        raise FeatureAuditError("domain-agnostic audit must not attach adaptation provenance")
    manifest_dir = Path(manifest_dir)
    feature_root = Path(feature_root)
    temporal_root = Path(temporal_root)
    split_reports: dict[str, Any] = {}
    audit_records: list[dict[str, Any]] = []

    for protocol_split, cache_split in split_dirs.items():
        manifest_path = manifest_dir / f"{protocol_split}.jsonl"
        manifest = _read_manifest(manifest_path)
        expected = {str(record["sample_id"]): record for record in manifest}
        feature_paths = _inventory_ids(feature_root / cache_split, ".pkl")
        metadata_paths = _inventory_ids(feature_root / cache_split, ".pkl.meta.json")
        support_paths = _inventory_ids(temporal_root / cache_split, ".json")
        expected_ids = set(expected)
        inventory_errors = {
            "missing_features": sorted(expected_ids - set(feature_paths)),
            "extra_features": sorted(set(feature_paths) - expected_ids),
            "missing_metadata": sorted(expected_ids - set(metadata_paths)),
            "extra_metadata": sorted(set(metadata_paths) - expected_ids),
            "missing_supports": sorted(expected_ids - set(support_paths)),
            "extra_supports": sorted(set(support_paths) - expected_ids),
        }
        common = sorted(
            expected_ids & set(feature_paths) & set(metadata_paths) & set(support_paths)
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(
                executor.map(
                    lambda sample_id: _audit_one(
                        expected[sample_id],
                        feature_paths[sample_id],
                        metadata_paths[sample_id],
                        support_paths[sample_id],
                        expected_checkpoint_sha256,
                        expected_stream_name,
                    ),
                    common,
                )
            )
        audit_records.extend(records)
        failures = [record for record in records if record["status"] != "PASS"]
        has_inventory_error = any(inventory_errors.values())
        split_reports[protocol_split] = {
            "status": "PASS" if not has_inventory_error and not failures else "FAIL_TECHNICAL",
            "cache_split": cache_split,
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
                "records": len(manifest),
            },
            "audited_records": len(records),
            "total_windows": sum(record["windows"] for record in records),
            "feature_bytes": sum(record["feature_bytes"] for record in records),
            "recipe_sha256": sorted({record["recipe_sha256"] for record in records}),
            "inventory_errors": {key: value[:100] for key, value in inventory_errors.items()},
            "inventory_error_counts": {key: len(value) for key, value in inventory_errors.items()},
            "failure_count": len(failures),
            "failure_examples": failures[:100],
        }

    compact_records = [
        {
            key: record[key]
            for key in (
                "sample_id",
                "feature_sha256",
                "metadata_sha256",
                "support_sha256",
                "source_video_sha256",
                "shape",
            )
        }
        for record in sorted(audit_records, key=lambda item: item["sample_id"])
    ]
    passed = all(report["status"] == "PASS" for report in split_reports.values())
    return {
        "schema_version": "ocem.feature_cache_audit.v1",
        "status": "PASS" if passed else "FAIL_TECHNICAL",
        "stream": expected_stream_name,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "feature_root": str(feature_root.resolve()),
        "temporal_root": str(temporal_root.resolve()),
        "split_reports": split_reports,
        "sample_artifact_digest": canonical_json_sha256(compact_records),
        "sample_artifact_count": len(compact_records),
        "raw_content_rehashed": False,
        "raw_verification_basis": (
            "manifest path/size/frame-count/FPS plus extractor-recorded video SHA-256; "
            "raw video bytes were not rehashed by this audit"
        ),
        "adaptation_provenance": adaptation_provenance,
        "feature_lock_eligible": passed and adaptation_provenance is not None,
        "feature_lock_blocker": (
            None
            if passed and adaptation_provenance is not None
            else (
                "Only the Oxford domain-agnostic stream was audited. A P14T train-only adapted "
                "checkpoint and aligned adapted stream are still required by WP-04."
                if expected_stream_name == "domain_agnostic"
                else "The adapted cache audit did not pass every artifact check."
            )
        ),
    }
