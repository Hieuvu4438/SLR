from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Method1Config
from .utils import sha256_file


class AuditError(ValueError):
    pass


def _file_check(path: str, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise AuditError(f"{label} is missing: {source}")
    return {"path": str(source.resolve()), "bytes": source.stat().st_size, "sha256": sha256_file(source)}


def audit_resources(config: Method1Config, stage: str) -> dict[str, Any]:
    if stage not in {"input", "method", "inference"}:
        raise AuditError(f"unsupported audit stage: {stage}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "ready",
        "stage": stage,
        "config_sha256": config.digest,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "resources": {},
    }
    resources = report["resources"]
    resources["clip_checkpoint"] = _file_check(
        config.model.clip_checkpoint_path, label="CLIP initialization"
    )
    resources["bpe_vocabulary"] = _file_check(config.model.bpe_path, label="BPE vocabulary")
    if stage == "input":
        resources["source_annotations"] = {
            split: _file_check(path, label=f"{split} source annotation")
            for split, path in config.data.source_annotations.items()
        }
        for name, root in (
            ("agnostic_root", config.data.agnostic_root),
            ("aware_root", config.data.aware_root),
        ):
            directory = Path(root)
            if not directory.is_dir():
                raise AuditError(f"{name} is missing: {directory}")
            resources[name] = str(directory.resolve())
        membership = Path(config.data.official_membership_json)
        resources["official_membership"] = (
            _file_check(str(membership), label="official membership")
            if membership.is_file()
            else {"status": "derivable", "configured_sources": config.data.official_split_annotations}
        )
    elif stage == "method":
        manifest_dir = Path(config.data.manifest_dir)
        for name in ("texts.jsonl", "videos.jsonl", "groups.jsonl", "splits.json", "resources.json", "manifest_meta.json"):
            resources[name] = _file_check(str(manifest_dir / name), label=name)
        resources["reference_checkpoint"] = _file_check(
            config.reference.checkpoint, label="selected baseline/reference checkpoint"
        )
        cache_meta = Path(config.reference.cache_dir) / "cache_meta.json"
        resources["reference_cache"] = _file_check(str(cache_meta), label="reference cache metadata")
    else:
        # Inference/export intentionally excludes reference, miner, and training captions.
        manifest_dir = Path(config.data.manifest_dir)
        for name in ("texts.jsonl", "videos.jsonl", "groups.jsonl", "splits.json", "resources.json"):
            resources[name] = _file_check(str(manifest_dir / name), label=name)
    return report
