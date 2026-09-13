from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import Method1Config
from .utils import atomic_json_dump, sha256_file, sha256_json


class SelectionLockError(RuntimeError):
    pass


def _lock_path(config: Method1Config) -> Path:
    return Path(config.output.root) / "selection_lock.json"


def create_selection_lock(
    config: Method1Config, checkpoint: str | Path
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    expected_path = (Path(config.output.root) / "best_dev.pt").resolve()
    if checkpoint_path != expected_path:
        raise SelectionLockError("selection lock requires this run's best_dev.pt")
    destination = _lock_path(config)
    if destination.exists():
        raise SelectionLockError(f"refusing to overwrite selection lock: {destination}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise SelectionLockError(f"cannot load selected checkpoint: {error}") from error
    if (
        not payload.get("training_run_complete", False)
        or payload.get("dev_selection") is None
        or payload.get("dev_metrics") is None
    ):
        raise SelectionLockError("selection lock rejects incomplete, pilot, or non-selected weights")
    if payload.get("config_sha256") != config.digest:
        raise SelectionLockError("selected checkpoint config identity differs")
    dev_metrics = payload["dev_metrics"]
    if dev_metrics.get("split") not in {None, "dev"}:
        raise SelectionLockError("selection checkpoint metrics are not dev metrics")
    report = {
        "schema_version": 1,
        "status": "locked",
        "policy": "dev_selected_configuration_v1",
        "dataset": config.data.dataset,
        "seed": config.seed,
        "arm": payload.get("arm"),
        "config_sha256": config.digest,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dev_selection": payload["dev_selection"],
        "dev_metrics_sha256": sha256_json(dev_metrics),
        "implementation_revision": payload.get("implementation_revision"),
        "artifact_hashes_sha256": sha256_json(payload.get("artifact_hashes")),
        "test_metrics_observed": False,
    }
    report["content_sha256"] = sha256_json(report)
    atomic_json_dump(report, destination)
    report["path"] = str(destination.resolve())
    return report


def validate_selection_lock(
    config: Method1Config,
    checkpoint: str | Path,
    *,
    allow_test_observed: bool = False,
) -> dict[str, Any]:
    path = _lock_path(config)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionLockError(f"test/export selection lock is missing or invalid: {error}") from error
    content_sha256 = report.get("content_sha256")
    payload = dict(report)
    payload.pop("content_sha256", None)
    if content_sha256 != sha256_json(payload):
        raise SelectionLockError("selection lock content hash mismatch")
    expected = {
        "status": "locked",
        "policy": "dev_selected_configuration_v1",
        "dataset": config.data.dataset,
        "seed": config.seed,
        "config_sha256": config.digest,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    if not allow_test_observed:
        expected["test_metrics_observed"] = False
    mismatches = {
        key: {"lock": report.get(key), "expected": value}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise SelectionLockError(f"selection lock identity mismatch: {mismatches}")
    return report


def record_test_observation(
    config: Method1Config,
    checkpoint: str | Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    report = validate_selection_lock(config, checkpoint)
    if metrics.get("split") != "test" or metrics.get("status") != "complete":
        raise SelectionLockError("only a completed test report can consume the selection lock")
    report["test_metrics_observed"] = True
    report["test_metrics_sha256"] = sha256_json(metrics)
    report["test_checkpoint_sha256"] = sha256_file(checkpoint)
    if metrics.get("metrics_path"):
        report["test_metrics_artifact_sha256"] = sha256_file(metrics["metrics_path"])
    report.pop("content_sha256", None)
    report["content_sha256"] = sha256_json(report)
    atomic_json_dump(report, _lock_path(config))
    return report
