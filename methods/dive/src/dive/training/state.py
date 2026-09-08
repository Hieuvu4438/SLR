from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


class CheckpointError(ValueError):
    """Checkpoint contents or provenance do not match the requested run."""


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    best_dev: dict[str, Any]
    sampler_state: dict[str, Any]
    fingerprints: dict[str, str]
    config_hash: str


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise CheckpointError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise CheckpointError("checkpoint contains CUDA RNG but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise CheckpointError("checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(cuda_state)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_dev: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    config_hash: str,
    git_revision: str,
    optimizer_manifest: Mapping[str, Any],
) -> str:
    if epoch < 0 or global_step < 0 or not config_hash or not git_revision:
        raise CheckpointError("checkpoint counters/config/git revision are invalid")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": "dive_checkpoint.v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_dev": dict(best_dev),
        "rng": capture_rng_state(),
        "sampler_state": dict(sampler_state),
        "fingerprints": dict(fingerprints),
        "config_hash": config_hash,
        "git_revision": git_revision,
        "optimizer_manifest": dict(optimizer_manifest),
        "torch_version": torch.__version__,
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    checksum = _checksum(destination)
    checksum_path = destination.with_suffix(destination.suffix + ".sha256")
    checksum_tmp = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    checksum_tmp.write_text(checksum + "\n", encoding="ascii")
    os.replace(checksum_tmp, checksum_path)
    return checksum


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    expected_config_hash: str,
    expected_fingerprints: Mapping[str, str],
    restore_rng: bool = True,
) -> ResumeState:
    source = Path(path)
    checksum_path = source.with_suffix(source.suffix + ".sha256")
    if not source.is_file() or not checksum_path.is_file():
        raise CheckpointError(f"checkpoint or checksum sidecar missing: {source}")
    expected_checksum = checksum_path.read_text(encoding="ascii").strip()
    if _checksum(source) != expected_checksum:
        raise CheckpointError("checkpoint checksum mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "dive_checkpoint.v1":
        raise CheckpointError("unsupported checkpoint schema")
    if payload.get("config_hash") != expected_config_hash:
        raise CheckpointError("checkpoint config hash mismatch")
    if payload.get("fingerprints") != dict(expected_fingerprints):
        raise CheckpointError("checkpoint data/bank/reference fingerprint mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if (scheduler is None) != (payload["scheduler"] is None):
        raise CheckpointError("checkpoint scheduler presence differs")
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if (scaler is None) != (payload["scaler"] is None):
        raise CheckpointError("checkpoint scaler presence differs")
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return ResumeState(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        best_dev=dict(payload["best_dev"]),
        sampler_state=dict(payload["sampler_state"]),
        fingerprints=dict(payload["fingerprints"]),
        config_hash=str(payload["config_hash"]),
    )
