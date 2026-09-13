from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import Method1Config


CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(value: Any, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def dev_selection_key(metrics: Mapping[str, Any], optimizer_step: int) -> tuple[float, float, int]:
    try:
        mean_r1 = 0.5 * (float(metrics["T2V"]["R1"]) + float(metrics["V2T"]["R1"]))
        mean_r5 = 0.5 * (float(metrics["T2V"]["R5"]) + float(metrics["V2T"]["R5"]))
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError("dev metrics must contain finite T2V/V2T R1 and R5") from error
    if not np.isfinite([mean_r1, mean_r5]).all() or optimizer_step < 0:
        raise CheckpointError("dev selector received nonfinite metrics or a negative step")
    return mean_r1, mean_r5, -int(optimizer_step)


@dataclass(frozen=True)
class DevSelection:
    optimizer_step: int
    mean_bidirectional_r1: float
    mean_bidirectional_r5: float

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any], optimizer_step: int) -> "DevSelection":
        r1, r5, _ = dev_selection_key(metrics, optimizer_step)
        return cls(optimizer_step, r1, r5)

    @property
    def key(self) -> tuple[float, float, int]:
        return self.mean_bidirectional_r1, self.mean_bidirectional_r5, -self.optimizer_step

    def beats(self, previous: "DevSelection | None") -> bool:
        return previous is None or self.key > previous.key


def make_training_checkpoint(
    *,
    config: Method1Config,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    sampler_state: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
    implementation_revision: str,
    dev_metrics: Mapping[str, Any] | None,
    scaler: Any | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if min(epoch, next_batch_index, global_step) < 0:
        raise CheckpointError("checkpoint progress fields must be non-negative")
    selection = (
        DevSelection.from_metrics(dev_metrics, global_step) if dev_metrics is not None else None
    )
    resolved_config = asdict(config)
    resolved_config.pop("source_path", None)
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method_version": config.method_version,
        "baseline_version": config.baseline_version,
        "upstream_commit": config.upstream_commit,
        "implementation_revision": implementation_revision,
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "global_step": int(global_step),
        "sampler_state": dict(sampler_state),
        "rng_state": dict(rng_state) if rng_state is not None else capture_rng_state(),
        "resolved_config": resolved_config,
        "config_sha256": config.digest,
        "artifact_hashes": dict(artifact_hashes),
        "arm": config.auxiliary.arm,
        "dev_selection": asdict(selection) if selection is not None else None,
        "dev_metrics": dict(dev_metrics) if dev_metrics is not None else None,
    }


def validate_resume_identity(
    checkpoint: Mapping[str, Any],
    *,
    config: Method1Config,
    artifact_hashes: Mapping[str, Any],
) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method_version": config.method_version,
        "baseline_version": config.baseline_version,
        "upstream_commit": config.upstream_commit,
        "config_sha256": config.digest,
        "arm": config.auxiliary.arm,
    }
    mismatches = {
        key: {"checkpoint": checkpoint.get(key), "expected": value}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if checkpoint.get("artifact_hashes") != dict(artifact_hashes):
        mismatches["artifact_hashes"] = {
            "checkpoint": checkpoint.get("artifact_hashes"),
            "expected": dict(artifact_hashes),
        }
    if mismatches:
        raise CheckpointError(f"resume identity mismatch: {mismatches}")
