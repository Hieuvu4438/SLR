from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elsc.config import config_hash
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
