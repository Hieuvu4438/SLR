"""Explicit local resource verification and lock generation."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class ResourceVerificationError(RuntimeError):
    """Raised for malformed verifier instructions, not ordinary missing assets."""


def _verify_delimited_text(path: Path, loader: Mapping[str, Any]) -> dict[str, Any]:
    delimiter = str(loader.get("delimiter", "\t"))
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return {"status": "FAIL", "reason": "empty annotation file"}
        rows = sum(1 for _ in reader)
    required_header = [str(item) for item in loader.get("header_contains", [])]
    missing_header = sorted(set(required_header) - set(header))
    expected_rows = loader.get("expected_rows")
    passed = not missing_header and (expected_rows is None or rows == int(expected_rows))
    return {
        "status": "PASS" if passed else "FAIL",
        "header": header,
        "missing_header": missing_header,
        "rows": rows,
        "expected_rows": expected_rows,
    }


def _verify_directory(path: Path, loader: Mapping[str, Any]) -> dict[str, Any]:
    missing_paths = [
        relative for relative in loader.get("required_paths", []) if not (path / relative).exists()
    ]
    file_counts = {}
    mismatches = {}
    for pattern, expected in loader.get("file_counts", {}).items():
        actual = sum(1 for candidate in path.glob(pattern) if candidate.is_file())
        file_counts[pattern] = actual
        if actual != int(expected):
            mismatches[pattern] = {"expected": int(expected), "actual": actual}
    passed = not missing_paths and not mismatches
    return {
        "status": "PASS" if passed else "FAIL",
        "missing_paths": missing_paths,
        "file_counts": file_counts,
        "file_count_mismatches": mismatches,
    }


def _verify_git(path: Path, loader: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(loader["expected_commit"])
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip() if completed.returncode == 0 else None
    missing_paths = [
        relative for relative in loader.get("required_paths", []) if not (path / relative).is_file()
    ]
    passed = completed.returncode == 0 and actual == expected and not missing_paths
    return {
        "status": "PASS" if passed else "FAIL",
        "expected_commit": expected,
        "actual_commit": actual,
        "missing_paths": missing_paths,
        "worktree_clean_required": False,
    }


def _state_dict_summary(state: Mapping[str, Any], loader: Mapping[str, Any]) -> dict[str, Any]:
    keys = list(state)
    required = [str(item) for item in loader.get("required_state_keys", [])]
    missing = sorted(set(required) - set(keys))
    min_tensors = int(loader.get("min_state_tensors", 1))
    passed = not missing and len(keys) >= min_tensors
    shapes = {}
    for key in required:
        if key in state and hasattr(state[key], "shape"):
            shapes[key] = list(state[key].shape)
    return {
        "status": "PASS" if passed else "FAIL",
        "state_tensors": len(keys),
        "minimum_state_tensors": min_tensors,
        "missing_state_keys": missing,
        "required_state_shapes": shapes,
    }


def _verify_torch(path: Path, loader: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    kind = loader["kind"]
    if kind == "torchscript":
        model = torch.jit.load(str(path), map_location="cpu").eval()
        result = _state_dict_summary(model.state_dict(), loader)
        result["format"] = "torchscript"
        return result
    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    if kind == "torch_state_dict":
        if not isinstance(loaded, Mapping):
            return {"status": "FAIL", "reason": "checkpoint is not a state dict mapping"}
        result = _state_dict_summary(loaded, loader)
        result["format"] = "torch_state_dict"
        return result
    if not isinstance(loaded, Mapping):
        return {"status": "FAIL", "reason": "checkpoint is not a mapping"}
    required_top = [str(item) for item in loader.get("required_top_level_keys", [])]
    missing_top = sorted(set(required_top) - set(loaded))
    state_key = str(loader.get("state_key", "state_dict"))
    state = loaded.get(state_key)
    if missing_top or not isinstance(state, Mapping):
        return {
            "status": "FAIL",
            "missing_top_level_keys": missing_top,
            "state_key": state_key,
            "state_mapping_present": isinstance(state, Mapping),
        }
    result = _state_dict_summary(state, loader)
    result.update(
        {
            "format": "torch_checkpoint",
            "top_level_keys": sorted(str(key) for key in loaded),
            "missing_top_level_keys": missing_top,
        }
    )
    return result


def verify_resource(resource: Mapping[str, Any]) -> dict[str, Any]:
    resource_id = str(resource["id"])
    required = bool(resource["required"])
    excluded = bool(resource.get("excluded_dependency", False))
    result: dict[str, Any] = {
        "id": resource_id,
        "kind": resource["kind"],
        "required": required,
        "excluded_dependency": excluded,
        "source_url": resource.get("source_url"),
        "configured_access_status": resource.get("access_status"),
    }
    if excluded:
        result.update({"status": "PASS_EXCLUDED", "local_path": None, "loaded": False})
        return result
    path = Path(str(resource["local_path"])).resolve()
    result["local_path"] = str(path)
    if not path.exists():
        result.update({"status": "FAIL_MISSING" if required else "OPTIONAL_MISSING", "loaded": False})
        return result
    result["path_type"] = "directory" if path.is_dir() else "file"
    if path.is_file():
        result["bytes"] = path.stat().st_size
        expected_bytes = resource.get("expected_bytes")
        result["expected_bytes"] = expected_bytes
        if expected_bytes is not None and result["bytes"] != int(expected_bytes):
            result.update({"status": "FAIL_SIZE", "loaded": False})
            return result
        expected_sha256 = resource.get("expected_sha256")
        if expected_sha256:
            result["sha256"] = sha256_file(path)
            result["expected_sha256"] = expected_sha256
            if result["sha256"] != expected_sha256:
                result.update({"status": "FAIL_HASH", "loaded": False})
                return result
    loader = resource.get("loader") or {"kind": "directory" if path.is_dir() else "file"}
    loader_kind = loader.get("kind")
    try:
        if loader_kind == "directory":
            content = _verify_directory(path, loader)
        elif loader_kind == "git":
            content = _verify_git(path, loader)
        elif loader_kind == "delimited_text":
            content = _verify_delimited_text(path, loader)
        elif loader_kind in {"torch_checkpoint", "torch_state_dict", "torchscript"}:
            content = _verify_torch(path, loader)
        elif loader_kind == "file":
            content = {"status": "PASS"}
        else:
            raise ResourceVerificationError(f"{resource_id}: unsupported loader kind {loader_kind!r}")
    except Exception as error:
        result.update(
            {
                "status": "FAIL_CONTENT",
                "loaded": False,
                "content": {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)},
            }
        )
        return result
    result["content"] = content
    result["loaded"] = loader_kind in {"torch_checkpoint", "torch_state_dict", "torchscript"}
    result["status"] = "PASS" if content["status"] == "PASS" else "FAIL_CONTENT"
    return result


def build_resource_lock(config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    results = [verify_resource(resource) for resource in config["resources"]]
    blocking = [
        item["id"]
        for item in results
        if item["required"] and item["status"] not in {"PASS", "PASS_EXCLUDED"}
    ]
    body: dict[str, Any] = {
        "schema_version": "ocem.resource_lock.v1",
        "status": "PASS" if not blocking else "BLOCKED_RESOURCE",
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": canonical_json_sha256(config),
        "resources": results,
        "blocking_resource_ids": blocking,
        "seds_dependency": False,
        "scientific_gate_promoted": False,
        "note": "This lock verifies registered files/content. Dataset protocol and feature locks remain separate G0 requirements.",
    }
    body["lock_sha256"] = canonical_json_sha256(body)
    return body
