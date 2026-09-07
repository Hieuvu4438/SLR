from __future__ import annotations

import json
from pathlib import Path

import pytest

from elsc.config import config_hash
from elsc.export import _validated_checkpoint
from elsc.provenance import ProvenanceError
from elsc.utils import sha256_file


def test_export_accepts_only_frozen_dev_selected_checkpoint(tmp_path: Path):
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    selected = checkpoints / "best_dev.pt"
    selected.write_bytes(b"selected")
    last = checkpoints / "last.pt"
    last.write_bytes(b"last")
    dev_manifest = tmp_path / "dev.jsonl"
    dev_manifest.write_text("{}\n", encoding="utf-8")
    config = {"data": {"dev_manifest": str(dev_manifest)}}
    (run / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": "dev",
                "test_used_for_selection": False,
                "checkpoint": "checkpoints/best_dev.pt",
                "checkpoint_sha256": sha256_file(selected),
                "config_hash": config_hash(config),
                "dev_manifest_sha256": sha256_file(dev_manifest),
            }
        ),
        encoding="utf-8",
    )
    assert _validated_checkpoint(run, "best_dev", config) == selected
    with pytest.raises(ProvenanceError, match="selection record"):
        _validated_checkpoint(run, "last", config)
