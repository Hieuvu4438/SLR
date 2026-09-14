from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from pmgr.config import config_hash
from slr_common.utils import sha256_file


IMPLEMENTATION_PATHS = (
    "conftest.py",
    "methods/README.md",
    "methods/pmgr",
    "pyproject.toml",
)
RESOURCE_KEYS = (
    "train_manifest",
    "validation_manifest",
    "train_index",
    "validation_index",
    "baseline_resolved_args",
    "initialization_checkpoint",
)
DEPENDENCIES = (
    "torch",
    "torchvision",
    "numpy",
    "PyYAML",
    "ftfy",
    "regex",
    "tqdm",
)


def _git(root: Path, *arguments: str, binary: bool = False):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def dependency_versions() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in DEPENDENCIES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def implementation_identity(root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    head = _git(repository, "rev-parse", "HEAD").strip()
    diff = _git(
        repository,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *IMPLEMENTATION_PATHS,
        binary=True,
    )
    return {
        "git_commit": head,
        "implementation_paths": list(IMPLEMENTATION_PATHS),
        "implementation_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "implementation_dirty": bool(diff),
    }


def resource_fingerprints(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in RESOURCE_KEYS:
        path = Path(config["paths"][name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required PMGR resource is unavailable: {path}")
        output[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def build_resume_invariants(
    config: dict[str, Any], *, repository_root: str | Path, world_size: int
) -> dict[str, Any]:
    cico_root = Path(config["paths"]["cico_root"]).resolve()
    actual_upstream = _git(cico_root, "rev-parse", "HEAD").strip()
    if actual_upstream != config["upstream_commit"]:
        raise ValueError(
            f"CiCo checkout is {actual_upstream}, expected {config['upstream_commit']}"
        )
    return {
        "schema_version": 1,
        "config_sha256": config_hash(config),
        "implementation": implementation_identity(repository_root),
        "upstream_commit": actual_upstream,
        "resources": resource_fingerprints(config),
        "dependencies": dependency_versions(),
        "world_size": int(world_size),
        "precision": config["engine"]["precision"],
        "deterministic": {
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    }
