from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

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
