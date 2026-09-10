"""Consolidate independently audited I3D streams into one dataset feature lock."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class FeatureLockError(ValueError):
    """Raised when two cache audits cannot prove aligned feature streams."""


def _load(path: Path, expected_sha256: str, stream: str) -> Mapping[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise FeatureLockError(f"{stream} audit SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureLockError(f"invalid {stream} audit report: {error}") from error
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != "ocem.feature_cache_audit.v1"
        or value.get("status") != "PASS"
        or value.get("stream") != stream
    ):
        raise FeatureLockError(f"{stream} audit is not a matching PASS report")
    return value


def build_phoenix_feature_lock(
    *,
    agnostic_audit: str | Path,
    expected_agnostic_audit_sha256: str,
    adapted_audit: str | Path,
    expected_adapted_audit_sha256: str,
) -> dict[str, Any]:
    """Require ID/support/recipe alignment before promoting P14T G0 resources."""

    agnostic_path, adapted_path = Path(agnostic_audit), Path(adapted_audit)
    agnostic = _load(
        agnostic_path, expected_agnostic_audit_sha256, "domain_agnostic"
    )
    adapted = _load(
        adapted_path, expected_adapted_audit_sha256, "domain_adapted_p14t"
    )
    if adapted.get("feature_lock_eligible") is not True or not isinstance(
        adapted.get("adaptation_provenance"), Mapping
    ):
        raise FeatureLockError("adapted audit lacks eligible train-only provenance")
    if agnostic.get("temporal_root") != adapted.get("temporal_root"):
        raise FeatureLockError("feature streams do not share one temporal-support root")
    if agnostic.get("sample_artifact_count") != adapted.get("sample_artifact_count"):
        raise FeatureLockError("feature stream sample counts differ")
    agnostic_splits, adapted_splits = agnostic.get("split_reports"), adapted.get(
        "split_reports"
    )
    if not isinstance(agnostic_splits, Mapping) or not isinstance(adapted_splits, Mapping):
        raise FeatureLockError("audit split reports are missing")
    if set(agnostic_splits) != {"train", "validation", "test"} or set(
        adapted_splits
    ) != set(agnostic_splits):
        raise FeatureLockError("audits do not contain the three protocol splits")

    alignment: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        left, right = agnostic_splits[split], adapted_splits[split]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise FeatureLockError(f"invalid split report for {split}")
        fields = {
            "manifest_sha256": (left["manifest"]["sha256"], right["manifest"]["sha256"]),
            "manifest_records": (left["manifest"]["records"], right["manifest"]["records"]),
            "audited_records": (left["audited_records"], right["audited_records"]),
            "total_windows": (left["total_windows"], right["total_windows"]),
            "recipe_sha256": (left["recipe_sha256"], right["recipe_sha256"]),
        }
        mismatched = [name for name, values in fields.items() if values[0] != values[1]]
        if mismatched:
            raise FeatureLockError(f"stream alignment failed for {split}: {mismatched}")
        alignment[split] = {
            "manifest_sha256": fields["manifest_sha256"][0],
            "samples": fields["manifest_records"][0],
            "windows": fields["total_windows"][0],
            "recipe_sha256": fields["recipe_sha256"][0],
        }
    return {
        "schema_version": "ocem.feature_lock.v1",
        "status": "PASS",
        "dataset": "phoenix2014t",
        "streams": {
            "domain_agnostic": {
                "root": agnostic["feature_root"],
                "checkpoint_sha256": agnostic["checkpoint_sha256"],
                "audit": {
                    "path": str(agnostic_path.resolve()),
                    "sha256": expected_agnostic_audit_sha256,
                },
                "sample_artifact_digest": agnostic["sample_artifact_digest"],
            },
            "domain_adapted_p14t": {
                "root": adapted["feature_root"],
                "checkpoint_sha256": adapted["checkpoint_sha256"],
                "audit": {
                    "path": str(adapted_path.resolve()),
                    "sha256": expected_adapted_audit_sha256,
                },
                "sample_artifact_digest": adapted["sample_artifact_digest"],
                "adaptation_provenance": adapted["adaptation_provenance"],
            },
        },
        "temporal_root": agnostic["temporal_root"],
        "alignment": alignment,
        "alignment_sha256": canonical_json_sha256(alignment),
        "sample_count": agnostic["sample_artifact_count"],
        "total_windows": sum(item["windows"] for item in alignment.values()),
        "validation_or_test_used_for_training": False,
        "ready_for_dataset_g0": True,
    }
