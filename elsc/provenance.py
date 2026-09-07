from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elsc.config import config_hash, load_config
from elsc.utils import sha256_file


class ProvenanceError(ValueError):
    pass


def validate_dev_selection(record_path: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    record_file = Path(record_path)
    checkpoint_file = Path(checkpoint_path)
    if not record_file.is_file():
        raise ProvenanceError(f"teacher selection record does not exist: {record_file}")
    if not checkpoint_file.is_file():
        raise ProvenanceError(f"teacher checkpoint does not exist: {checkpoint_file}")
    value = json.loads(record_file.read_text(encoding="utf-8"))
    if value.get("selection_split") != "dev" or value.get("test_used_for_selection") is not False:
        raise ProvenanceError("teacher must be selected on dev without test feedback")
    actual_hash = sha256_file(checkpoint_file)
    if value.get("checkpoint_sha256") != actual_hash:
        raise ProvenanceError("teacher checkpoint hash differs from its dev selection record")
    return value


def validate_test_lock(
    record_path: str | Path,
    checkpoint_path: str | Path,
    *,
    run_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate every frozen dev-selection input before touching the test gallery."""
    value = validate_dev_selection(record_path, checkpoint_path)
    run_root = Path(run_dir).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    try:
        relative_checkpoint = checkpoint.relative_to(run_root).as_posix()
    except ValueError as error:
        raise ProvenanceError(
            "test checkpoint must be inside the selected run directory"
        ) from error
    if value.get("checkpoint") != relative_checkpoint:
        raise ProvenanceError("requested test checkpoint is not the dev-selected checkpoint")
    if value.get("config_hash") != config_hash(config):
        raise ProvenanceError("resolved config differs from the frozen dev selection")
    dev_manifest = Path(config["data"]["dev_manifest"])
    if value.get("dev_manifest_sha256") != sha256_file(dev_manifest):
        raise ProvenanceError("dev manifest differs from the frozen dev selection")
    return value


def validate_completed_run(
    run_dir: str | Path, *, expected_config_path: str | Path | None = None
) -> dict[str, Any]:
    run_root = Path(run_dir)
    config_path = run_root / "resolved_config.yaml"
    summary_path = run_root / "run_summary.json"
    selection_path = run_root / "selection.json"
    if not config_path.is_file() or not summary_path.is_file() or not selection_path.is_file():
        raise ProvenanceError("run is missing resolved config, summary, or selection")
    config = load_config(config_path, validate=False)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint_value = selection.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ProvenanceError("run selection lacks a checkpoint path")
    checkpoint = run_root / checkpoint_value
    validate_test_lock(selection_path, checkpoint, run_dir=run_root, config=config)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ProvenanceError("run summary status is not complete")
    if summary.get("selection_sha256") != sha256_file(selection_path):
        raise ProvenanceError("run summary does not hash the current selection")
    if expected_config_path is not None:
        expected = load_config(expected_config_path, validate=False)
        if config_hash(config) != config_hash(expected):
            raise ProvenanceError("completed run config differs from the expected config")
    return {
        "schema_version": 1,
        "status": "complete_validated",
        "run_dir": str(run_root.resolve()),
        "config_hash": config_hash(config),
        "selection_sha256": sha256_file(selection_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_epoch": selection.get("selected_epoch"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a completed dev-selected ELSC training run"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    result = validate_completed_run(
        args.run_dir,
        expected_config_path=args.config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
