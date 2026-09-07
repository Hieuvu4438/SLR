from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch


class ResourceGuardError(RuntimeError):
    pass


def resource_snapshot(path: str | Path, device: torch.device) -> dict[str, Any]:
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    result: dict[str, Any] = {
        "disk_probe_path": str(probe),
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "device": str(device),
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(gpu_free_bytes=int(free), gpu_total_bytes=int(total))
    else:
        result.update(gpu_free_bytes=None, gpu_total_bytes=None)
    return result


def require_resources(
    path: str | Path,
    device: torch.device,
    *,
    min_disk_gib: float,
    min_gpu_gib: float,
    operation: str,
) -> dict[str, Any]:
    snapshot = resource_snapshot(path, device)
    gib = 1024**3
    if snapshot["disk_free_bytes"] < float(min_disk_gib) * gib:
        raise ResourceGuardError(
            f"{operation} requires at least {min_disk_gib:g} GiB free disk; "
            f"found {snapshot['disk_free_bytes'] / gib:.2f} GiB"
        )
    gpu_free = snapshot["gpu_free_bytes"]
    if gpu_free is not None and gpu_free < float(min_gpu_gib) * gib:
        raise ResourceGuardError(
            f"{operation} requires at least {min_gpu_gib:g} GiB free GPU memory; "
            f"found {gpu_free / gib:.2f} GiB"
        )
    return snapshot


def require_storage_budget(
    path: str | Path,
    *,
    planned_write_bytes: int,
    min_remaining_gib: float,
    operation: str,
) -> dict[str, Any]:
    """Reject a write plan that could consume the configured disk reserve.

    ``require_resources`` protects the reserve at process start.  Feature
    extraction additionally knows an upper bound for its output, so this guard
    accounts for that write before any model or video is loaded.
    """

    if planned_write_bytes < 0:
        raise ValueError("planned_write_bytes must be non-negative")
    snapshot = resource_snapshot(path, torch.device("cpu"))
    gib = 1024**3
    remaining = snapshot["disk_free_bytes"] - int(planned_write_bytes)
    required = float(min_remaining_gib) * gib
    if remaining < required:
        raise ResourceGuardError(
            f"{operation} plans to write up to {planned_write_bytes / gib:.2f} GiB, "
            f"which would leave {remaining / gib:.2f} GiB free; "
            f"the required reserve is {min_remaining_gib:g} GiB"
        )
    snapshot["planned_write_bytes"] = int(planned_write_bytes)
    snapshot["projected_disk_free_bytes"] = int(remaining)
    return snapshot
