from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from elsc.config import config_hash, load_config
from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator, encode_cico_text
from elsc.provenance import ProvenanceError, validate_dev_selection
from elsc.upstream.factory import build_retriever_from_checkpoint, load_cico_tokenizer
from slr_common.utils import atomic_json_dump, sha256_file


def _resolve(root: Path, value: object) -> Path | None:
    if value is None or not isinstance(value, str) or value.lower().startswith("required"):
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _asset(path: Path | None, *, hash_file: bool = False) -> dict[str, Any]:
    if path is None:
        return {"status": "unresolved"}
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": path.exists()}
    if not path.exists():
        result["status"] = "missing"
    elif path.is_file():
        result.update(status="ready", bytes=path.stat().st_size)
        if hash_file:
            result["sha256"] = sha256_file(path)
    else:
        result.update(status="ready", kind="directory")
    return result


def _pickle_count(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    with path.open("rb") as handle:
        value = pickle.load(handle)
    return len(value) if hasattr(value, "__len__") else None


def _pickle_keys(path: Path | None) -> set[str] | None:
    if path is None or not path.is_file():
        return None
    with path.open("rb") as handle:
        value = pickle.load(handle)
    return {str(key) for key in value} if isinstance(value, dict) else None


def _official_keys(path: Path | None, *, delimiter: str = "|") -> set[str] | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row["name"]) for row in csv.DictReader(handle, delimiter=delimiter)}


def audit_assets(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    root = (
        config_path.resolve().parent.parent if config_path.parent.name == "configs" else Path.cwd()
    )
    paths: dict[str, dict[str, Any]] = {}
    sources = config.get("sources", {})
    prepared_splits = tuple(sources.get("prepared_splits", ("train", "dev", "test")))
    if not prepared_splits or len(set(prepared_splits)) != len(prepared_splits):
        raise ValueError("sources.prepared_splits must be non-empty and unique")
    invalid_splits = set(prepared_splits) - {"train", "dev", "test"}
    if invalid_splits:
        raise ValueError(f"invalid prepared splits: {sorted(invalid_splits)}")
    official_delimiter = str(sources.get("official_annotation_delimiter", "|"))
    for key in (
        "train_annotation",
        "dev_annotation",
        "test_annotation",
        "train_official_annotation",
        "dev_official_annotation",
        "test_official_annotation",
        "feature_agnostic_root",
        "feature_aware_root",
        "temporal_metadata_root",
    ):
        paths[f"sources.{key}"] = _asset(
            _resolve(root, sources.get(key)), hash_file="annotation" in key
        )
    model = config.get("model", {})
    for key in ("init_checkpoint", "teacher_checkpoint"):
        asset = _asset(_resolve(root, model.get(key)), hash_file=True)
        expected_hash = model.get(f"{key}_sha256")
        if expected_hash is not None:
            asset["expected_sha256"] = expected_hash
            asset["hash_match"] = asset.get("sha256") == expected_hash
        paths[f"model.{key}"] = asset
    upstream = config.get("upstream", {})
    cico_root = _resolve(root, upstream.get("cico_root"))
    paths["upstream.cico_root"] = _asset(cico_root)
    actual_commit = None
    if cico_root is not None and cico_root.exists():
        repository = cico_root
        while repository != repository.parent and not (repository / ".git").exists():
            repository = repository.parent
        if (repository / ".git").exists():
            actual_commit = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    expected_commit = upstream.get("cico_commit")
    disk = shutil.disk_usage(root)
    counts = {
        split: _pickle_count(_resolve(root, sources.get(f"{split}_annotation")))
        for split in ("train", "dev", "test")
    }
    feature_split_integrity: dict[str, Any] = {}
    for source_key in ("feature_agnostic_root", "feature_aware_root"):
        label = f"sources.{source_key}"
        feature_root = _resolve(root, sources.get(source_key))
        if feature_root is None or not feature_root.is_dir():
            continue
        stream_complete = True
        stream_splits: dict[str, Any] = {}
        for split in prepared_splits:
            official = _official_keys(
                _resolve(root, sources.get(f"{split}_official_annotation")),
                delimiter=official_delimiter,
            )
            actual = {path.stem for path in (feature_root / split).glob("*.pkl")}
            missing_ids = sorted((official or set()) - actual)
            extra_ids = sorted(actual - (official or set())) if official is not None else []
            complete = official is not None and not missing_ids and not extra_ids
            stream_complete &= complete
            stream_splits[split] = {
                "expected_count": len(official) if official is not None else None,
                "actual_count": len(actual),
                "missing_count": len(missing_ids),
                "extra_count": len(extra_ids),
                "missing_ids_preview": missing_ids[:20],
                "extra_ids_preview": extra_ids[:20],
                "complete": complete,
            }
        feature_split_integrity[source_key] = stream_splits
        paths[label]["split_integrity"] = stream_splits
        if not stream_complete:
            paths[label]["status"] = "incomplete"
    required_keys = {
        "sources.train_annotation",
        "sources.dev_annotation",
        "sources.test_annotation",
        "sources.feature_agnostic_root",
        "sources.feature_aware_root",
        "model.init_checkpoint",
        "upstream.cico_root",
    }
    for split in prepared_splits:
        if sources.get(f"{split}_official_annotation") is not None:
            required_keys.add(f"sources.{split}_official_annotation")
    required_keys = {
        key
        for key in required_keys
        if not key.startswith("sources.")
        or not key.endswith("_annotation")
        or any(key == f"sources.{split}_annotation" for split in prepared_splits)
        or any(key == f"sources.{split}_official_annotation" for split in prepared_splits)
    }
    if config.get("method") != "baseline":
        required_keys.add("model.teacher_checkpoint")
    if config.get("evidence", {}).get("enabled"):
        required_keys.add("sources.temporal_metadata_root")
    unresolved = sorted(
        key
        for key in required_keys
        if paths.get(key, {"status": "unresolved"})["status"] == "unresolved"
    )
    missing = sorted(
        key
        for key in required_keys
        if paths.get(key, {"status": "missing"})["status"] in {"missing", "incomplete"}
    )
    hash_mismatches = sorted(
        key for key in required_keys if paths.get(key, {}).get("hash_match") is False
    )
    commit_ok = actual_commit == expected_commit
    split_integrity: dict[str, Any] = {}
    official_sets: dict[str, set[str]] = {}
    for split in prepared_splits:
        source_keys = _pickle_keys(_resolve(root, sources.get(f"{split}_annotation")))
        official_keys = _official_keys(
            _resolve(root, sources.get(f"{split}_official_annotation")),
            delimiter=official_delimiter,
        )
        if source_keys is not None and official_keys is not None:
            official_sets[split] = official_keys
            split_integrity[split] = {
                "official_count": len(official_keys),
                "all_official_ids_in_source": official_keys <= source_keys,
                "source_extra_count": len(source_keys - official_keys),
                "source_missing_count": len(official_keys - source_keys),
            }
    cross_split_overlap: dict[str, int] = {}
    split_names = sorted(official_sets)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            cross_split_overlap[f"{left}__{right}"] = len(
                official_sets[left] & official_sets[right]
            )
    expected_counts = {"train": 7096, "dev": 519, "test": 642}
    split_integrity_ok = all(
        details["source_missing_count"] == 0
        and (
            config.get("data", {}).get("dataset") != "ph"
            or details["official_count"] == expected_counts[split]
        )
        for split, details in split_integrity.items()
    ) and not any(cross_split_overlap.values())
    return {
        "schema_version": 1,
        "status": (
            "ready"
            if not unresolved
            and not missing
            and not hash_mismatches
            and commit_ok
            and split_integrity_ok
            else "blocked_assets"
        ),
        "config_hash": config_hash(config),
        "paths": paths,
        "annotation_counts": counts,
        "diagnostics": {
            "ph_dev_source_looks_train_plus_dev": counts.get("train") == 7096
            and counts.get("dev") == 7615,
            "official_ph_expected_counts": expected_counts,
            "split_integrity": split_integrity,
            "cross_split_official_id_overlap": cross_split_overlap,
            "split_integrity_ok": split_integrity_ok,
            "feature_split_integrity": feature_split_integrity,
            "prepared_splits": list(prepared_splits),
        },
        "upstream": {
            "expected_commit": expected_commit,
            "actual_commit": actual_commit,
            "match": commit_ok,
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "safe_for_large_generation": disk.free >= 100 * 1024**3,
            "configured_min_free_bytes": int(
                float(config.get("resources", {}).get("min_free_disk_gib", 20)) * 1024**3
            ),
            "safe_for_configured_operations": disk.free
            >= float(config.get("resources", {}).get("min_free_disk_gib", 20)) * 1024**3,
        },
        "unresolved": unresolved,
        "missing": missing,
        "hash_mismatches": hash_mismatches,
    }


def audit_teacher(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    root = config_path.resolve().parent.parent
    checkpoint = _resolve(root, config.get("model", {}).get("teacher_checkpoint"))
    provenance = config.get("model", {}).get("teacher_selection_provenance")
    asset = _asset(checkpoint, hash_file=True)
    expected_hash = config.get("model", {}).get("teacher_checkpoint_sha256")
    if expected_hash is not None:
        asset["expected_sha256"] = expected_hash
        asset["hash_match"] = asset.get("sha256") == expected_hash
    provenance_path = _resolve(root, provenance)
    provenance_asset = _asset(provenance_path, hash_file=True)
    error = None
    if checkpoint is not None and provenance_path is not None:
        try:
            validate_dev_selection(provenance_path, checkpoint)
        except ProvenanceError as exception:
            error = str(exception)
    ready = (
        asset.get("status") == "ready"
        and asset.get("hash_match", True)
        and provenance_asset.get("status") == "ready"
        and error is None
    )
    return {
        "schema_version": 1,
        "status": "ready" if ready else "blocked_assets",
        "teacher_checkpoint": asset,
        "provenance": provenance_asset,
        "validation_error": error,
    }


def audit_checkpoint(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Exercise the real release checkpoint without claiming dataset-level golden parity."""
    assets = audit_assets(config, config_path)
    needed = ["upstream.cico_root", "model.init_checkpoint"]
    missing = [key for key in needed if assets["paths"][key]["status"] != "ready"]
    if assets.get("hash_mismatches"):
        missing.extend(assets["hash_mismatches"])
    if missing:
        return {
            "schema_version": 1,
            "status": "blocked_assets",
            "missing_or_invalid": sorted(set(missing)),
        }
    root = config_path.resolve().parent.parent
    checkpoint = _resolve(root, config["model"]["init_checkpoint"])
    assert checkpoint is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_retriever_from_checkpoint(config, checkpoint, device=device)
    model.eval()
    tokenizer = load_cico_tokenizer(config)
    encoded = [
        encode_cico_text(text, tokenizer, int(config["data"]["max_words"]))
        for text in ("weather tomorrow", "heavy rain possible")
    ]
    ids, segments, input_mask = (
        torch.stack([item[index] for item in encoded]).to(device) for index in range(3)
    )
    generator = torch.Generator().manual_seed(20260907)
    h = torch.randn(2, int(config["data"]["feature_len"]), 1024, generator=generator).to(device)
    valid = torch.ones(h.shape[:2], dtype=torch.bool, device=device)
    valid[1, -13:] = False
    h[1, -13:] = 0
    with torch.inference_mode():
        bridged_video = model.bridge.encode_video(h, valid)
        bridged_text = model.bridge.encode_text(ids, segments, input_mask)
        direct_video = model.core.get_visual_output(
            model.bridge.upstream_video(h),
            model.bridge.upstream_video_mask(valid),
            shaped=True,
            video_frame=1,
            get_hidden=True,
        )
        direct_text = model.core.get_sequence_output(
            ids, segments, input_mask, shaped=False, get_hidden=True
        )
        bridge_i2t, bridge_t2i = model.bridge.score(bridged_video, bridged_text, objective=True)
        direct_i2t, direct_t2i, *_ = model.core.get_similarity_logits(
            direct_text[1],
            direct_video[1],
            direct_text[0],
            direct_video[0],
            shaped=True,
            loose_type=model.core.loose_type,
            is_train=True,
        )
    comparisons = {
        "video_tokens": float((bridged_video.tokens - direct_video[1]).abs().max()),
        "text_tokens": float((bridged_text.tokens - direct_text[1]).abs().max()),
        "i2t": float((bridge_i2t - direct_i2t).abs().max()),
        "t2i": float((bridge_t2i - direct_t2i).abs().max()),
    }
    mixed = model.bridge.mixed_score(bridge_i2t, bridge_t2i, float(config["model"]["dual_mix"]))
    passed = all(error <= 1e-6 for error in comparisons.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "checkpoint_smoke_fixture.pt"
    torch.save(
        {
            "schema_version": 1,
            "scope": "synthetic_input_component_smoke_only",
            "checkpoint_sha256": sha256_file(checkpoint),
            "seed": 20260907,
            "h": h.cpu(),
            "valid": valid.cpu(),
            "input_ids": ids.cpu(),
            "input_mask": input_mask.cpu(),
            "mixed": mixed.cpu(),
        },
        fixture_path,
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": "synthetic-input component smoke; not dataset-level golden parity",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "atol": 1e-6,
        "max_abs_errors": comparisons,
        "fixture": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
    }


def audit_parity(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    assets = audit_assets(config, config_path)
    needed = ["upstream.cico_root", "model.init_checkpoint"]
    missing = [key for key in needed if assets["paths"][key]["status"] != "ready"]
    manifest = Path(config["data"]["dev_manifest"])
    if not manifest.is_file():
        missing.append("data.dev_manifest")
    if missing:
        return {"schema_version": 1, "status": "blocked_assets", "missing": missing}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(config["model"]["init_checkpoint"])
    model, _ = build_retriever_from_checkpoint(config, checkpoint, device=device)
    model.eval()
    tokenizer = load_cico_tokenizer(config)
    dataset = CiCoFeatureDataset(
        manifest,
        feature_len=int(config["data"]["feature_len"]),
        alpha=float(config["data"]["alpha"]),
        split="dev",
    )
    loader = DataLoader(
        dataset,
        batch_size=min(4, len(dataset)),
        shuffle=False,
        collate_fn=CiCoCollator(tokenizer, int(config["data"]["max_words"]), augment=False),
    )
    batch = next(iter(loader))
    h = batch["h"].to(device)
    valid = batch["valid"].to(device)
    ids, segments, input_mask = (value.to(device) for value in batch["clean_text"])
    with torch.no_grad():
        bridged_video = model.bridge.encode_video(h, valid)
        bridged_text = model.bridge.encode_text(ids, segments, input_mask)
        upstream_mask = model.bridge.upstream_video_mask(valid)
        direct_video = model.core.get_visual_output(
            model.bridge.upstream_video(h),
            upstream_mask,
            shaped=True,
            video_frame=1,
            get_hidden=True,
        )
        direct_text = model.core.get_sequence_output(
            ids, segments, input_mask, shaped=False, get_hidden=True
        )
        bridge_i2t, bridge_t2i = model.bridge.score(bridged_video, bridged_text, objective=True)
        direct_i2t, direct_t2i, *_ = model.core.get_similarity_logits(
            direct_text[1],
            direct_video[1],
            direct_text[0],
            direct_video[0],
            shaped=True,
            loose_type=model.core.loose_type,
            is_train=True,
        )
        bridge_mixed = model.bridge.mixed_score(
            bridge_i2t, bridge_t2i, float(config["model"]["dual_mix"])
        )
        direct_mixed = model.bridge.mixed_score(
            direct_i2t, direct_t2i, float(config["model"]["dual_mix"])
        )
    comparisons = {
        "video_tokens": float((bridged_video.tokens - direct_video[1]).abs().max()),
        "text_tokens": float((bridged_text.tokens - direct_text[1]).abs().max()),
        "i2t": float((bridge_i2t - direct_i2t).abs().max()),
        "t2i": float((bridge_t2i - direct_t2i).abs().max()),
        "mixed": float((bridge_mixed - direct_mixed).abs().max()),
    }
    ranks_equal = torch.equal(
        torch.argsort(bridge_mixed, dim=1, descending=True),
        torch.argsort(direct_mixed, dim=1, descending=True),
    ) and torch.equal(
        torch.argsort(bridge_mixed.T, dim=1, descending=True),
        torch.argsort(direct_mixed.T, dim=1, descending=True),
    )
    passed = all(error <= 1e-6 for error in comparisons.values()) and ranks_equal
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "golden_fixture.pt"
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_sha256": sha256_file(checkpoint),
            "pair_ids": batch["pair_id"],
            "video_ids": batch["video_id"],
            "caption_ids": batch["caption_id"],
            "h": batch["h"],
            "valid": batch["valid"],
            "dense_index": batch["dense_index"],
            "input_ids": batch["clean_text"][0],
            "input_mask": batch["clean_text"][2],
            "video_tokens": bridged_video.tokens.cpu(),
            "text_tokens": bridged_text.tokens.cpu(),
            "i2t": bridge_i2t.cpu(),
            "t2i": bridge_t2i.cpu(),
            "mixed": bridge_mixed.cpu(),
            "v2t_order": torch.argsort(bridge_mixed, dim=1, descending=True).cpu(),
            "t2v_order": torch.argsort(bridge_mixed.T, dim=1, descending=True).cpu(),
        },
        fixture_path,
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": "component-level pinned CiCo encoder/scorer parity",
        "dtype": str(h.dtype),
        "atol": 1e-6,
        "max_abs_errors": comparisons,
        "ranks_equal": ranks_equal,
        "fixture": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit ELSC dependencies and stage gates without downloading assets"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=("assets", "teacher", "checkpoint", "parity"), required=True
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path, stage="assets")
    if args.stage == "assets":
        result = audit_assets(config, config_path)
        default = "artifacts/asset_audit.json"
    elif args.stage == "teacher":
        result = audit_teacher(config, config_path)
        default = "artifacts/teacher_audit.json"
    elif args.stage == "checkpoint":
        default = "artifacts/parity/checkpoint_smoke.json"
        requested_output = Path(args.output or default)
        checkpoint_dir = (
            requested_output if requested_output.suffix == "" else requested_output.parent
        )
        result = audit_checkpoint(config, config_path, checkpoint_dir)
    else:
        default = "artifacts/parity/parity_report.json"
        requested_output = Path(args.output or default)
        parity_dir = requested_output if requested_output.suffix == "" else requested_output.parent
        result = audit_parity(config, config_path, parity_dir)
    output = Path(args.output or default)
    if args.stage in {"checkpoint", "parity"} and output.suffix == "":
        report_name = (
            "checkpoint_smoke.json" if args.stage == "checkpoint" else "parity_report.json"
        )
        output = output / report_name
    atomic_json_dump(result, output)
    if result["status"] == "blocked_assets":
        atomic_json_dump(result, "artifacts/blocked_assets.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ready", "passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
