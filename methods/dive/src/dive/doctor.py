from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import config_hash, required_resource_paths


@dataclass(frozen=True)
class ResourceCheck:
    field: str
    path: str | None
    exists: bool
    error_code: str | None


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    torch_info: dict[str, Any] = {"installed": False}
    try:
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "gpus": [],
        }
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                torch_info["gpus"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
    except Exception as exc:  # pragma: no cover - environment-specific diagnostic
        torch_info["error"] = f"{type(exc).__name__}: {exc}"

    checks: list[ResourceCheck] = []
    for field, raw_path in required_resource_paths(config, stage).items():
        value = None if raw_path is None else os.path.expandvars(os.path.expanduser(str(raw_path)))
        exists = value is not None and Path(value).exists()
        checks.append(
            ResourceCheck(
                field=field,
                path=value,
                exists=exists,
                error_code=None if exists else _resource_error_code(field),
            )
        )
    missing = [asdict(item) for item in checks if not item.exists]
    return {
        "schema_version": "doctor.v1",
        "stage": stage,
        "ready": not missing,
        "config_hash": config_hash(config),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _version("numpy"),
            "pyyaml": _version("PyYAML"),
            "torch": torch_info,
        },
        "resources": [asdict(item) for item in checks],
        "missing": missing,
    }


def _resource_error_code(field: str) -> str:
    if "dev_manifest" in field:
        return "MISSING_DEV_ARTIFACT"
    if "test_manifest" in field:
        return "MISSING_TEST_ARTIFACT"
    if "train_manifest" in field:
        return "MISSING_TRAIN_ARTIFACT"
    if "tokenizer" in field:
        return "UNVERIFIED_TEXT_MAPPING"
    if "audit" in field:
        return "MISSING_SCHEMA_AUDIT"
    if "checkpoint" in field or "weights" in field:
        return "MISSING_BASELINE_CHECKPOINT"
    if "experiment_plan" in field:
        return "MISSING_LOCKED_EXPERIMENT_PLAN"
    return "MISSING_RESOURCE"


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
