from __future__ import annotations

import hashlib
import json

import pytest

from ocem.data.feature_lock import FeatureLockError, build_phoenix_feature_lock


def _write(path, value):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _audit(stream):
    split_reports = {}
    for split, count, windows in (
        ("train", 7, 70),
        ("validation", 2, 20),
        ("test", 3, 30),
    ):
        split_reports[split] = {
            "status": "PASS",
            "manifest": {"sha256": split * 8, "records": count},
            "audited_records": count,
            "total_windows": windows,
            "recipe_sha256": ["recipe"],
        }
    return {
        "schema_version": "ocem.feature_cache_audit.v1",
        "status": "PASS",
        "stream": stream,
        "feature_root": f"/features/{stream}",
        "temporal_root": "/supports",
        "checkpoint_sha256": stream * 4,
        "sample_artifact_count": 12,
        "sample_artifact_digest": stream * 3,
        "split_reports": split_reports,
        "feature_lock_eligible": stream == "domain_adapted_p14t",
        "adaptation_provenance": (
            {"sha256": "adapted", "fixed_epoch": 15}
            if stream == "domain_adapted_p14t"
            else None
        ),
    }


def test_feature_lock_requires_and_records_exact_stream_alignment(tmp_path) -> None:
    agnostic_path, adapted_path = tmp_path / "a.json", tmp_path / "b.json"
    agnostic_sha = _write(agnostic_path, _audit("domain_agnostic"))
    adapted_sha = _write(adapted_path, _audit("domain_adapted_p14t"))
    result = build_phoenix_feature_lock(
        agnostic_audit=agnostic_path,
        expected_agnostic_audit_sha256=agnostic_sha,
        adapted_audit=adapted_path,
        expected_adapted_audit_sha256=adapted_sha,
    )
    assert result["status"] == "PASS"
    assert result["ready_for_dataset_g0"] is True
    assert result["sample_count"] == 12
    assert result["total_windows"] == 120


def test_feature_lock_rejects_window_mismatch(tmp_path) -> None:
    agnostic, adapted = _audit("domain_agnostic"), _audit("domain_adapted_p14t")
    adapted["split_reports"]["test"]["total_windows"] += 1
    agnostic_path, adapted_path = tmp_path / "a.json", tmp_path / "b.json"
    agnostic_sha = _write(agnostic_path, agnostic)
    adapted_sha = _write(adapted_path, adapted)
    with pytest.raises(FeatureLockError, match="alignment failed"):
        build_phoenix_feature_lock(
            agnostic_audit=agnostic_path,
            expected_agnostic_audit_sha256=agnostic_sha,
            adapted_audit=adapted_path,
            expected_adapted_audit_sha256=adapted_sha,
        )
