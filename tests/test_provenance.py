from __future__ import annotations

import json
from pathlib import Path

import pytest

from elsc.config import config_hash
from elsc.provenance import ProvenanceError, validate_dev_selection, validate_test_lock
from elsc.utils import sha256_file


def test_teacher_must_match_dev_selection_record(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_split": "dev",
                "test_used_for_selection": False,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    assert validate_dev_selection(selection, checkpoint)["selection_split"] == "dev"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ProvenanceError, match="hash differs"):
        validate_dev_selection(selection, checkpoint)


def test_test_selected_teacher_is_rejected(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_split": "test",
                "test_used_for_selection": True,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="selected on dev"):
        validate_dev_selection(selection, checkpoint)


def test_test_lock_rejects_nonselected_checkpoint_and_changed_inputs(tmp_path: Path):
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    selected = checkpoints / "best_dev.pt"
    selected.write_bytes(b"selected")
    other = checkpoints / "last.pt"
    other.write_bytes(b"other")
    dev = tmp_path / "dev.jsonl"
    dev.write_text("dev\n", encoding="utf-8")
    config = {"data": {"dev_manifest": str(dev)}}
    selection = run / "selection.json"
    record = {
        "selection_split": "dev",
        "test_used_for_selection": False,
        "checkpoint": "checkpoints/best_dev.pt",
        "checkpoint_sha256": sha256_file(selected),
        "config_hash": config_hash(config),
        "dev_manifest_sha256": sha256_file(dev),
    }
    selection.write_text(json.dumps(record), encoding="utf-8")
    assert validate_test_lock(selection, selected, run_dir=run, config=config) == record

    record["checkpoint_sha256"] = sha256_file(other)
    selection.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not the dev-selected"):
        validate_test_lock(selection, other, run_dir=run, config=config)

    record["checkpoint"] = "checkpoints/last.pt"
    selection.write_text(json.dumps(record), encoding="utf-8")
    changed_config = {"data": {"dev_manifest": str(dev)}, "changed": True}
    with pytest.raises(ProvenanceError, match="config differs"):
        validate_test_lock(selection, other, run_dir=run, config=changed_config)

    record["config_hash"] = config_hash(changed_config)
    selection.write_text(json.dumps(record), encoding="utf-8")
    dev.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="dev manifest differs"):
        validate_test_lock(selection, other, run_dir=run, config=changed_config)
