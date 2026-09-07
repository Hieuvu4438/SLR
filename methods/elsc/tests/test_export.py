from __future__ import annotations

import json
from pathlib import Path

import pytest

from elsc.config import config_hash
from elsc.export import _inference_config, _validated_checkpoint
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


def test_inference_config_removes_training_only_dependencies():
    config = {
        "schema_version": 1,
        "sources": {"feature_agnostic_root": "/features"},
        "data": {
            "feature_dim": 1024,
            "feature_len": 64,
            "max_words": 32,
            "train_manifest": "/train",
            "dev_manifest": "/dev",
            "test_manifest": "/test",
            "text_augmentation": "cico_random_swap_v1",
        },
        "model": {
            "init_checkpoint": "/init",
            "teacher_checkpoint": "/teacher",
            "teacher_checkpoint_sha256": "teacher-hash",
            "teacher_selection_provenance": "/selection",
            "lexical_head": {"output_dim": 512},
            "adapter": {"enabled": True},
        },
        "cache": {"path": "/cache"},
        "mining": {"mass_min": 0.6},
        "loss": {"lexical_weight": 0.1},
        "evidence": {"enabled": True},
        "keep": {"weight": 0.0},
        "caption": {"weight": 0.0},
        "train": {"epochs": 20},
    }
    result = _inference_config(config, source_checkpoint_sha256="checkpoint-hash")
    for key in ("sources", "cache", "mining", "loss", "evidence", "keep", "caption", "train"):
        assert key not in result
    for key in ("train_manifest", "dev_manifest", "test_manifest", "text_augmentation"):
        assert key not in result["data"]
    for key in ("init_checkpoint", "teacher_checkpoint", "lexical_head"):
        assert key not in result["model"]
    assert result["model"]["adapter"] == {"enabled": True}
    assert result["export_provenance"]["source_checkpoint_sha256"] == "checkpoint-hash"
