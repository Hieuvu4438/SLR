"""Read-only environment diagnostics."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_doctor_report(probe_path: str | Path = ".") -> dict[str, Any]:
    """Collect facts without importing torch unless the doctor command is run."""

    path = Path(probe_path).resolve()
    usage = shutil.disk_usage(path)
    report: dict[str, Any] = {
        "schema_version": "ocem.doctor.v1",
        "status": "ANALYZED",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "probe_path": str(path),
        "disk": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "torch", "torchvision", "PyYAML")
        },
        "cuda": {"available": False, "devices": []},
        "blockers": [],
    }
    try:
        import torch

        available = torch.cuda.is_available()
        report["cuda"]["available"] = available
        report["cuda"]["torch_cuda"] = torch.version.cuda
        if available:
            report["cuda"]["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
        else:
            report["blockers"].append("CUDA is unavailable for GPU solver/training work packages.")
    except ImportError:
        report["blockers"].append("PyTorch is not installed; GPU solver/training cannot run.")
    return report

