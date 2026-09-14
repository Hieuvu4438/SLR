from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch

from pmgr.config import config_hash
from slr_common.utils import capture_rng_state, sha256_file


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    sampler_cursor: int,
    effective_step: int,
    best: dict[str, Any] | None,
    provenance: dict[str, Any],
    sampler_state: dict[str, Any],
    progress: dict[str, Any],
    validation_history: list[dict[str, Any]],
    resume_invariants: dict[str, Any],
) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "pmgr-training-v2",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "sampler_cursor": int(sampler_cursor),
        "effective_step": int(effective_step),
        "best": best,
        "rng": capture_rng_state(),
        "config": config,
        "config_hash": config_hash(config),
        "provenance": provenance,
        "sampler_state": sampler_state,
        "progress": progress,
        "validation_history": validation_history,
        "resume_invariants": resume_invariants,
        "schedule_state": {"effective_step": int(effective_step)},
        "resume_policy": "resume_exact",
    }
    required_bytes = math.ceil(_tensor_bytes(payload) * 1.15) + 16 * 1024**2
    free_bytes = os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize
    if free_bytes < required_bytes:
        raise OSError(
            f"insufficient storage for atomic PMGR checkpoint: need {required_bytes}, have {free_bytes}"
        )
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return {"path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size}


def validate_resume(
    raw: Any,
    config: dict[str, Any],
    *,
    resume_invariants: dict[str, Any] | None = None,
) -> None:
    if not isinstance(raw, dict) or raw.get("format") != "pmgr-training-v2":
        raise ValueError("exact resume requires a PMGR training checkpoint")
    if raw.get("config_hash") != config_hash(config):
        raise ValueError("resume config hash differs; use a new weights-only experiment")
    if raw.get("resume_policy") != "resume_exact" or "rng" not in raw:
        raise ValueError("checkpoint lacks exact-resume state")
    required = ("sampler_state", "progress", "validation_history", "resume_invariants")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError("checkpoint lacks ledger fields: " + ", ".join(missing))
    sampler = raw["sampler_state"]
    if (
        sampler.get("epoch") != raw.get("epoch")
        or sampler.get("cursor") != raw.get("sampler_cursor")
        or not isinstance(sampler.get("permutation"), list)
    ):
        raise ValueError("checkpoint sampler ledger is inconsistent")
    if raw["progress"].get("effective_steps") != raw.get("effective_step"):
        raise ValueError("checkpoint progress ledger is inconsistent")
    if resume_invariants is not None and raw["resume_invariants"] != resume_invariants:
        raise ValueError("resume invariant changed; start a new weights-only experiment")
