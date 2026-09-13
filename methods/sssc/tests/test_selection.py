from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from method1.config import load_config
from method1.selection import (
    SelectionLockError,
    create_selection_lock,
    record_test_observation,
    validate_selection_lock,
)


def _config(tmp_path: Path):
    base = load_config("methods/sssc/configs/method1/ph_local.yaml")
    return replace(base, output=replace(base.output, root=str(tmp_path / "run")))


def _checkpoint(config, *, complete: bool = True) -> Path:
    path = Path(config.output.root) / "best_dev.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "training_run_complete": complete,
            "dev_selection": {"mean_bidirectional_r1": 1.0},
            "dev_metrics": {"split": "dev", "T2V": {}, "V2T": {}},
            "config_sha256": config.digest,
            "arm": config.auxiliary.arm,
            "implementation_revision": "abc",
            "artifact_hashes": {"resources": "same"},
        },
        path,
    )
    return path


def test_selection_lock_binds_completed_best_dev_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = _checkpoint(config)
    report = create_selection_lock(config, checkpoint)
    assert report["test_metrics_observed"] is False
    validated = validate_selection_lock(config, checkpoint)
    assert validated["checkpoint_sha256"] == report["checkpoint_sha256"]
    with pytest.raises(SelectionLockError, match="overwrite"):
        create_selection_lock(config, checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(SelectionLockError, match="identity mismatch"):
        validate_selection_lock(config, checkpoint)


def test_selection_lock_rejects_pilot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = _checkpoint(config, complete=False)
    with pytest.raises(SelectionLockError, match="incomplete"):
        create_selection_lock(config, checkpoint)


def test_completed_test_consumes_lock_but_export_validation_remains_allowed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint = _checkpoint(config)
    create_selection_lock(config, checkpoint)
    consumed = record_test_observation(
        config,
        checkpoint,
        {"status": "complete", "split": "test", "T2V": {}, "V2T": {}},
    )
    assert consumed["test_metrics_observed"] is True
    with pytest.raises(SelectionLockError, match="identity mismatch"):
        validate_selection_lock(config, checkpoint)
    assert (
        validate_selection_lock(config, checkpoint, allow_test_observed=True)[
            "test_metrics_observed"
        ]
        is True
    )
