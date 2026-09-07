from __future__ import annotations

import json
from pathlib import Path

import pytest

from elsc.config import config_hash
from elsc.provenance import (
    ProvenanceError,
    validate_completed_run,
    validate_dev_selection,
    validate_test_lock,
)
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


def test_completed_run_validation_binds_summary_selection_and_expected_config(tmp_path: Path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "best_dev.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"selected")
    dev = tmp_path / "dev.jsonl"
    dev.write_text("{}\n", encoding="utf-8")
    config = {"schema_version": 1, "data": {"dev_manifest": str(dev)}}
    config_path = run / "resolved_config.yaml"
    config_path.write_text(
        f"schema_version: 1\ndata:\n  dev_manifest: {dev}\n", encoding="utf-8"
    )
    selection = {
        "selection_split": "dev",
        "test_used_for_selection": False,
        "checkpoint": "checkpoints/best_dev.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_hash": config_hash(config),
        "dev_manifest_sha256": sha256_file(dev),
        "selected_epoch": 2,
    }
    selection_path = run / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    summary_path = run / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {"status": "complete", "selection_sha256": sha256_file(selection_path)}
        ),
        encoding="utf-8",
    )
    result = validate_completed_run(run, expected_config_path=config_path)
    assert result["status"] == "complete_validated"
    assert result["selected_epoch"] == 2

    summary_path.write_text(
        json.dumps({"status": "complete", "selection_sha256": "stale"}),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="current selection"):
        validate_completed_run(run, expected_config_path=config_path)
