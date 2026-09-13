from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from method1.audit import AuditError, audit_resources
from method1.config import load_config


def test_input_audit_requires_every_feature_split_directory(tmp_path: Path) -> None:
    base = load_config("methods/sssc/configs/method1/ph_base_initial.yaml")
    annotations = {}
    for split in ("train", "dev", "test"):
        path = tmp_path / f"{split}.pkl"
        path.write_bytes(b"fixture")
        annotations[split] = str(path)
    agnostic = tmp_path / "agnostic"
    aware = tmp_path / "aware"
    for root in (agnostic, aware):
        (root / "train").mkdir(parents=True)
        (root / "dev").mkdir()
    config = replace(
        base,
        data=replace(
            base.data,
            source_annotations=annotations,
            official_membership_json=str(tmp_path / "membership.json"),
            official_split_annotations={},
            agnostic_root=str(agnostic),
            aware_root=str(aware),
        ),
    )
    with pytest.raises(AuditError, match="no established test feature directory"):
        audit_resources(config, "input")


def test_method_audit_scopes_auxiliary_cache_by_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("method1.audit.validate_manifest_bundle", lambda path: {"status": "ready"})
    base = load_config("methods/sssc/configs/method1/ph_base_initial.yaml")
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    for name in (
        "texts.jsonl",
        "videos.jsonl",
        "groups.jsonl",
        "splits.json",
        "resources.json",
        "manifest_meta.json",
    ):
        (manifest / name).write_text("fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "selected.pt"
    torch.save(
        {
            "arm": "base_initial",
            "dev_selection": {"mean_bidirectional_r1": 1.0},
            "training_run_complete": True,
        },
        checkpoint,
    )
    continuation = replace(
        base,
        data=replace(base.data, manifest_dir=str(manifest)),
        reference=replace(
            base.reference,
            checkpoint=str(checkpoint),
            cache_dir=str(tmp_path / "absent-cache"),
        ),
        auxiliary=replace(base.auxiliary, arm="base_continuation"),
    )
    report = audit_resources(continuation, "method")
    assert report["resources"]["auxiliary_cache"] == {
        "status": "not_required_for_base_continuation"
    }
    shared = replace(
        continuation,
        auxiliary=replace(continuation.auxiliary, arm="span_shared"),
    )
    with pytest.raises(AuditError, match="reference cache metadata"):
        audit_resources(shared, "method")
