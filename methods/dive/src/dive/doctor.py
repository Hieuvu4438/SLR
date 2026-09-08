from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactError, ArtifactResolver
from .config import config_hash, required_resource_paths


_PREPARED_PARENT_FIELDS = {
    "data.train_manifest": "train_manifest",
    "data.dev_manifest": "dev_manifest",
    "data.test_manifest": "test_manifest",
    "data.train_relations": "train_relations",
    "data.relevance_dir": "relevance_dir",
    "data.frame_maps_dir": "frame_maps_dir",
}
_CONFIG_PARENT_FIELDS = {
    **{
        field: ("prepare_data", output, "shared")
        for field, output in _PREPARED_PARENT_FIELDS.items()
    },
    "baseline.locked_checkpoint": ("baseline_train", "locked_checkpoint", "shared"),
}
_STAGE_PARENT_ARTIFACTS = {
    "baseline_train": (
        ("validate_data", "audit", "shared"),
        ("validate_data", "native_frame_maps_dir", "shared"),
    ),
    "baseline_validate": (
        ("validate_data", "audit", "shared"),
        ("validate_data", "native_frame_maps_dir", "shared"),
    ),
    "warmup": (
        ("validate_data", "audit", "shared"),
        ("validate_data", "native_frame_maps_dir", "shared"),
        ("validate_data", "text_unit_maps_dir", "shared"),
        ("baseline_validate", "report", "shared"),
    ),
    "mine_finalize": (
        ("baseline_validate", "report", "shared"),
        ("evidence_warmup", "reference", "shared"),
    ),
    "evaluate_test": (("validate_data", "audit", "shared"),),
}


@dataclass(frozen=True)
class ResourceCheck:
    field: str
    path: str | None
    exists: bool
    error_code: str | None
    resolution: str
    detail: str | None


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
    resolver = ArtifactResolver(config)
    for field, raw_path in required_resource_paths(config, stage).items():
        value = None
        detail = None
        resolution = "config"
        if raw_path is not None:
            value = os.path.expandvars(os.path.expanduser(str(raw_path)))
        elif field in _CONFIG_PARENT_FIELDS:
            parent_stage, parent_name, parent_scope = _CONFIG_PARENT_FIELDS[field]
            resolution = f"run_state:{parent_scope}/{parent_stage}/{parent_name}"
            try:
                parent = resolver.resolve(parent_stage, parent_name, scope=parent_scope)
                value = str(parent.path)
            except ArtifactError as exc:
                detail = str(exc)
        exists = value is not None and Path(value).exists()
        error_code = None
        if not exists:
            error_code = (
                "CACHE_HASH_MISMATCH"
                if detail is not None and "CACHE_HASH_MISMATCH" in detail
                else _resource_error_code(field)
            )
        checks.append(
            ResourceCheck(
                field=field,
                path=value,
                exists=exists,
                error_code=error_code,
                resolution=resolution,
                detail=detail,
            )
        )
    for parent_stage, name, scope in _STAGE_PARENT_ARTIFACTS.get(stage, ()):
        field = f"artifact.{parent_stage}.{name}"
        detail = None
        value = None
        try:
            parent = resolver.resolve(parent_stage, name, scope=scope)
            value = str(parent.path)
        except ArtifactError as exc:
            detail = str(exc)
        exists = value is not None and Path(value).exists()
        checks.append(
            ResourceCheck(
                field=field,
                path=value,
                exists=exists,
                error_code=(
                    None
                    if exists
                    else "CACHE_HASH_MISMATCH"
                    if detail is not None and "CACHE_HASH_MISMATCH" in detail
                    else "MISSING_PARENT_ARTIFACT"
                ),
                resolution=f"run_state:{scope}/{parent_stage}/{name}",
                detail=detail,
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
