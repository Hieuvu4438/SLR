from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .auxiliary_cache import Method1AuxiliaryCache
from .config import Method1Config
from .manifests import validate_manifest_bundle
from .reference_pipeline import reference_cache_identity
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
            split_directories = {}
            for split in ("train", "dev", "test"):
                split_directory = directory / split
                if not split_directory.is_dir():
                    raise AuditError(
                        f"{name} has no established {split} feature directory: {split_directory}"
                    )
                split_directories[split] = str(split_directory.resolve())
            resources[name] = {
                "path": str(directory.resolve()),
                "split_directories": split_directories,
            }
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
        resources["manifest_validation"] = validate_manifest_bundle(manifest_dir)
        resources["reference_checkpoint"] = _file_check(
            config.reference.checkpoint, label="selected baseline/reference checkpoint"
        )
        checkpoint = torch.load(
            config.reference.checkpoint, map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("arm") != "base_initial"
            or checkpoint.get("dev_selection") is None
            or not checkpoint.get("training_run_complete", False)
        ):
            raise AuditError(
                "method training requires a completed dev-selected base_initial checkpoint"
            )
        del checkpoint
        if config.auxiliary.arm != "base_continuation":
            cache_meta = Path(config.reference.cache_dir) / "cache_meta.json"
            resources["reference_cache"] = _file_check(
                str(cache_meta), label="reference cache metadata"
            )
            identity = reference_cache_identity(config)
            cache = Method1AuxiliaryCache(
                config.reference.cache_dir, expected_identity=identity
            )
            resources["auxiliary_cache"] = {
                "reference_identity_sha256": cache.reference_identity_sha256,
                "mining_content_sha256": cache.mining_content_sha256,
                "edit_count": len(cache.edits_by_uid),
            }
        else:
            resources["auxiliary_cache"] = {
                "status": "not_required_for_base_continuation"
            }
    else:
        # Inference/export intentionally excludes reference, miner, and training captions.
        manifest_dir = Path(config.data.manifest_dir)
        for name in ("texts.jsonl", "videos.jsonl", "groups.jsonl", "splits.json", "resources.json"):
            resources[name] = _file_check(str(manifest_dir / name), label=name)
        resources["manifest_meta.json"] = _file_check(
            str(manifest_dir / "manifest_meta.json"), label="manifest_meta.json"
        )
        resources["manifest_validation"] = validate_manifest_bundle(manifest_dir)
    return report
