from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ocem.data.features import FeatureAuditError, audit_feature_cache
from ocem.provenance.hashes import canonical_json_sha256, sha256_file


CHECKPOINT_SHA256 = "a" * 64


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    sample_id = "sample-1"
    manifest_dir = tmp_path / "manifests"
    feature_root = tmp_path / "features"
    temporal_root = tmp_path / "temporal"
    raw_root = tmp_path / "raw"
    for path in (manifest_dir, feature_root / "train", temporal_root / "train", raw_root):
        path.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / f"{sample_id}.mp4"
    raw_path.write_bytes(b"not-a-real-video-but-sized-and-identified")
    manifest = {
        "sample_id": sample_id,
        "raw_relpath": f"raw/{sample_id}.mp4",
        "raw_bytes": raw_path.stat().st_size,
        "frame_count": 18,
        "fps_num": 25,
        "fps_den": 1,
    }
    (manifest_dir / "train.jsonl").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    feature_path = feature_root / "train" / f"{sample_id}.pkl"
    with feature_path.open("wb") as handle:
        pickle.dump(
            {"name": str(raw_path), "feature": np.zeros((3, 1024), dtype=np.float32)},
            handle,
            protocol=5,
        )
    recipe = {"clip_frames": 16, "stride": 1}
    recipe_sha256 = canonical_json_sha256(recipe)
    source_sha256 = sha256_file(raw_path)
    metadata = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "feature_dtype": "float32",
        "feature_sha256": sha256_file(feature_path),
        "feature_shape": [3, 1024],
        "recipe_sha256": recipe_sha256,
        "source_video": str(raw_path),
        "source_video_sha256": source_sha256,
        "stream_name": "domain_agnostic",
        "schema_version": 1,
    }
    Path(f"{feature_path}.meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    support = {
        "coordinate_system": "input_frame",
        "decoded_frame_count": 18,
        "fps": 25.0,
        "interval_convention": "half_open",
        "recipe": recipe,
        "recipe_sha256": recipe_sha256,
        "rf_start": [0, 1, 2],
        "rf_end": [16, 17, 18],
        "source_video": str(raw_path),
        "source_video_sha256": source_sha256,
        "verified": True,
        "schema_version": 1,
    }
    (temporal_root / "train" / f"{sample_id}.json").write_text(
        json.dumps(support), encoding="utf-8"
    )
    return {
        "manifest_dir": manifest_dir,
        "feature_root": feature_root,
        "temporal_root": temporal_root,
    }


def _audit(paths: dict[str, Path]) -> dict:
    return audit_feature_cache(
        **paths,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        split_dirs={"train": "train"},
        workers=1,
    )


def test_feature_cache_audit_checks_payload_hash_shape_and_support(tmp_path) -> None:
    report = _audit(_write_fixture(tmp_path))
    assert report["status"] == "PASS"
    assert report["sample_artifact_count"] == 1
    assert report["split_reports"]["train"]["total_windows"] == 3
    assert report["feature_lock_eligible"] is False


def test_feature_cache_audit_rejects_support_drift(tmp_path) -> None:
    paths = _write_fixture(tmp_path)
    support_path = paths["temporal_root"] / "train" / "sample-1.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    support["rf_end"][-1] = 19
    support_path.write_text(json.dumps(support), encoding="utf-8")
    report = _audit(paths)
    assert report["status"] == "FAIL_TECHNICAL"
    errors = report["split_reports"]["train"]["failure_examples"][0]["errors"]
    assert "support_intervals_not_stride1_actual_frames" in errors


def test_feature_cache_audit_reports_missing_id_without_silent_drop(tmp_path) -> None:
    paths = _write_fixture(tmp_path)
    (paths["feature_root"] / "train" / "sample-1.pkl").unlink()
    report = _audit(paths)
    assert report["status"] == "FAIL_TECHNICAL"
    assert report["split_reports"]["train"]["audited_records"] == 0
    assert report["split_reports"]["train"]["inventory_errors"]["missing_features"] == ["sample-1"]
    assert report["split_reports"]["train"]["inventory_error_counts"]["missing_features"] == 1


def test_feature_cache_audit_rejects_disallowed_pickle_global(tmp_path) -> None:
    paths = _write_fixture(tmp_path)
    feature_path = paths["feature_root"] / "train" / "sample-1.pkl"
    with feature_path.open("wb") as handle:
        pickle.dump({"name": "irrelevant", "feature": eval}, handle, protocol=5)
    metadata_path = Path(f"{feature_path}.meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["feature_sha256"] = sha256_file(feature_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = _audit(paths)
    errors = report["split_reports"]["train"]["failure_examples"][0]["errors"]
    assert "feature_payload_invalid:UnpicklingError" in errors


def _adaptation_report(path: Path, *, leaked: bool = False) -> str:
    report = {
        "schema_version": "ocem.p14t_i3d_adaptation_run.v1",
        "status": "PASS",
        "completed_epochs": 15,
        "final_checkpoint": {"sha256": CHECKPOINT_SHA256, "fixed_epoch": 15},
        "config": {"checkpoint_selection": "fixed_final_epoch_15"},
        "data_policy": {
            "holdout_used_for_optimizer": False,
            "holdout_used_for_checkpoint_selection": False,
            "validation_or_test_used": leaked,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return sha256_file(path)


def test_adapted_cache_becomes_eligible_only_with_train_only_run_provenance(tmp_path) -> None:
    paths = _write_fixture(tmp_path)
    metadata_path = paths["feature_root"] / "train" / "sample-1.pkl.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stream_name"] = "domain_adapted_p14t"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report_path = tmp_path / "adaptation.json"
    report_sha256 = _adaptation_report(report_path)
    report = audit_feature_cache(
        **paths,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        split_dirs={"train": "train"},
        workers=1,
        expected_stream_name="domain_adapted_p14t",
        adaptation_report=report_path,
        expected_adaptation_report_sha256=report_sha256,
    )
    assert report["status"] == "PASS"
    assert report["feature_lock_eligible"] is True
    assert report["adaptation_provenance"]["fixed_epoch"] == 15


def test_adapted_cache_rejects_leaky_training_report(tmp_path) -> None:
    paths = _write_fixture(tmp_path)
    report_path = tmp_path / "adaptation.json"
    report_sha256 = _adaptation_report(report_path, leaked=True)
    with pytest.raises(FeatureAuditError, match="does not prove"):
        audit_feature_cache(
            **paths,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            split_dirs={"train": "train"},
            workers=1,
            expected_stream_name="domain_adapted_p14t",
            adaptation_report=report_path,
            expected_adaptation_report_sha256=report_sha256,
        )
