from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pmgr.config import config_hash, load_config
from pmgr.data.group_index import assert_split_disjoint, build_group_index, load_group_index
from slr_common.utils import atomic_json_dump, sha256_file


def _git(command: list[str], root: Path) -> str:
    result = subprocess.run(
        ["git", *command], cwd=root, check=True, text=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def _resource(path: str | None, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path:
        return {"path": path, "exists": False, "sha256": None, "matches": None}
    source = Path(path)
    actual = sha256_file(source) if source.is_file() else None
    return {
        "path": str(source),
        "exists": source.exists(),
        "sha256": actual,
        "matches": actual == expected_sha256 if actual and expected_sha256 else None,
    }


def audit(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path, mode="audit")
    root = Path.cwd().resolve()
    paths = config["paths"]
    cico_root = Path(paths["cico_root"])
    cico_git = cico_root
    try:
        actual_upstream = _git(["rev-parse", "HEAD"], cico_git)
    except (OSError, subprocess.CalledProcessError):
        actual_upstream = None
    resources = {
        name: _resource(paths.get(name), paths.get(f"{name}_sha256"))
        for name in (
            "train_manifest", "validation_manifest", "train_index", "validation_index",
            "baseline_resolved_args", "initialization_checkpoint",
        )
    }
    indexes: dict[str, Any] = {}
    loaded = []
    for split, key in (("train", "train_index"), ("validation", "validation_index")):
        path = paths.get(key)
        if path and Path(path).is_file():
            try:
                index = load_group_index(path, check_paths=True)
                loaded.append(index)
                indexes[split] = {
                    "status": "ready",
                    "split": index.split,
                    "groups": index.group_count,
                    "videos": index.video_count,
                }
            except ValueError as error:
                indexes[split] = {"status": "invalid", "error": str(error)}
        else:
            indexes[split] = {"status": "missing"}
    disjoint = None
    if len(loaded) == 2:
        try:
            assert_split_disjoint(*loaded)
            disjoint = True
        except ValueError as error:
            disjoint = False
            indexes["overlap_error"] = str(error)
    blocking = []
    if actual_upstream != config["upstream_commit"]:
        blocking.append("pinned upstream revision mismatch")
    for name, record in resources.items():
        if name not in {"train_manifest", "validation_manifest"} and not record["exists"]:
            blocking.append(f"missing {name}")
        if record["matches"] is False:
            blocking.append(f"{name} SHA-256 mismatch")
    if any(
        record.get("status") != "ready"
        for record in indexes.values()
        if isinstance(record, dict) and "status" in record
    ):
        blocking.append("canonical group indexes are not ready")
    if disjoint is False:
        blocking.append("train/validation video overlap")
    expected = {
        "train": (int(config["data"]["expected_train_groups"]), int(config["data"]["expected_train_videos"])),
        "validation": (int(config["data"]["expected_validation_groups"]), int(config["data"]["expected_validation_videos"])),
    }
    for split, (groups, videos) in expected.items():
        actual = indexes.get(split, {})
        if actual.get("status") == "ready" and (actual["groups"], actual["videos"]) != (groups, videos):
            blocking.append(f"{split} count mismatch")
    report = {
        "schema_version": 1,
        "status": "ready" if not blocking else "blocked",
        "completion_level": "software_ready_pending_tests" if not blocking else "inputs_incomplete",
        "config": str(Path(config_path).resolve()),
        "config_sha256": config_hash(config),
        "repository": {
            "root": str(root),
            "commit": _git(["rev-parse", "HEAD"], root),
            "branch": _git(["branch", "--show-current"], root),
            "dirty": bool(_git(["status", "--porcelain"], root)),
        },
        "upstream": {
            "expected_commit": config["upstream_commit"],
            "actual_commit": actual_upstream,
            "matches": actual_upstream == config["upstream_commit"],
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "numpy": np.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "resources": resources,
        "indexes": indexes,
        "split_disjoint": disjoint,
        "blocking": blocking,
    }
    if torch.cuda.is_available():
        report["runtime"]["gpu"] = torch.cuda.get_device_name(0)
    return report


def _build_indexes(config_path: str | Path, splits: list[str], skip_feature_check: bool) -> dict[str, Any]:
    config = load_config(config_path, mode="audit")
    paths = config["paths"]
    mapping = {
        "train": ("train_manifest", "train_index", "train"),
        "validation": ("validation_manifest", "validation_index", "dev"),
    }
    reports = {}
    for split in splits:
        manifest_key, index_key, expected_split = mapping[split]
        reports[split] = build_group_index(
            paths[manifest_key], paths[index_key], expected_split=expected_split,
            check_paths=not skip_feature_check,
        )
    if all(Path(paths[mapping[name][1]]).is_file() for name in mapping):
        assert_split_disjoint(
            load_group_index(paths["train_index"]), load_group_index(paths["validation_index"])
        )
    return {"schema_version": 1, "indexes": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PMGR runtime, data, and provenance commands")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--config", required=True)
    audit_parser.add_argument("--output")
    build = subparsers.add_parser("build-index")
    build.add_argument("--config", required=True)
    build.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    build.add_argument("--skip-feature-check", action="store_true")
    build.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit(args.config)
    else:
        result = _build_indexes(args.config, args.splits, args.skip_feature_check)
    if args.output:
        atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "ready") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
