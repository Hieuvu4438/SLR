from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from elsc.config import config_hash, load_config
from elsc.data.manifest import ManifestRecord, write_manifest
from elsc.utils import atomic_json_dump, ordered_hash, sha256_file


def _annotation_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        if "name" not in (reader.fieldnames or []):
            raise ValueError(f"official annotation lacks name column: {path}")
        return [str(row["name"]) for row in reader]


def _load_pickle(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"CiCo annotation must be a dictionary: {path}")
    return value


def _feature_shape(path: Path, feature_dim: int) -> tuple[int, int]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict):
        value = value.get("feature")
    shape = np.asarray(value).shape
    if len(shape) != 2:
        raise ValueError(f"feature at {path} is not rank-2: {shape}")
    if shape[0] == feature_dim and shape[1] == feature_dim:
        raise ValueError(f"feature orientation is ambiguous at {path}: {shape}")
    if shape[0] == feature_dim:
        return int(shape[1]), int(shape[0])
    if shape[1] == feature_dim:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"feature at {path} does not contain dim={feature_dim}: {shape}")


def prepare_split(
    config: dict[str, Any], split: str
) -> tuple[list[ManifestRecord], dict[str, Any]]:
    sources = config["sources"]
    data = config["data"]
    source_path = Path(sources[f"{split}_annotation"])
    annotations = _load_pickle(source_path)
    official_path = sources.get(f"{split}_official_annotation")
    if official_path:
        ordered_ids = _annotation_ids(official_path)
    else:
        ordered_ids = list(annotations)
    missing_annotation_ids = [
        identifier for identifier in ordered_ids if identifier not in annotations
    ]
    extras = sorted(set(annotations) - set(ordered_ids))
    aware_root = Path(sources["feature_aware_root"]) / split
    agnostic_root = Path(sources["feature_agnostic_root"]) / split
    records: list[ManifestRecord] = []
    missing_features: list[dict[str, str]] = []
    feature_dim = int(data["feature_dim"])
    for identifier in ordered_ids:
        if identifier not in annotations:
            continue
        item = annotations[identifier]
        video_name = str(item["video_name"])
        aware = aware_root / f"{video_name}.pkl"
        agnostic = agnostic_root / f"{video_name}.pkl"
        if not aware.is_file() or not agnostic.is_file():
            missing_features.append(
                {"pair_id": identifier, "aware": str(aware), "agnostic": str(agnostic)}
            )
            continue
        aware_shape = _feature_shape(aware, feature_dim)
        agnostic_shape = _feature_shape(agnostic, feature_dim)
        if aware_shape != agnostic_shape:
            raise ValueError(
                f"{identifier}: feature streams differ: {aware_shape} != {agnostic_shape}"
            )
        temporal_root = sources.get("temporal_metadata_root")
        temporal = (
            str(Path(temporal_root) / split / f"{video_name}.json") if temporal_root else None
        )
        records.append(
            ManifestRecord(
                schema_version=1,
                dataset=str(data["dataset"]),
                split=split,
                pair_id=identifier,
                video_id=video_name,
                caption_id=identifier,
                caption_original=str(item.get("ori_text", item["text"])),
                caption_model=str(item["text"]),
                caption_language=str(data["caption_language"]),
                feature_agnostic=str(agnostic.resolve()),
                feature_aware=str(aware.resolve()),
                dense_length=aware_shape[0],
                feature_dim=aware_shape[1],
                temporal_metadata=temporal,
            )
        )
    report = {
        "schema_version": 1,
        "split": split,
        "source_annotation": str(source_path),
        "source_annotation_sha256": sha256_file(source_path),
        "official_count": len(ordered_ids),
        "ordered_official_id_hash": ordered_hash(ordered_ids),
        "official_annotation_sha256": sha256_file(official_path) if official_path else None,
        "source_count": len(annotations),
        "ready_count": len(records),
        "source_extra_count": len(extras),
        "source_extra_ids_preview": extras[:20],
        "missing_annotation_ids": missing_annotation_ids,
        "missing_feature_count": len(missing_features),
        "missing_feature_ids": [item["pair_id"] for item in missing_features],
        "missing_features_preview": missing_features[:20],
    }
    return records, report


def cross_split_overlaps(
    records_by_split: dict[str, list[ManifestRecord]],
) -> dict[str, dict[str, list[str]]]:
    overlaps: dict[str, dict[str, list[str]]] = {}
    splits = sorted(records_by_split)
    for left_index, left_split in enumerate(splits):
        for right_split in splits[left_index + 1 :]:
            left = records_by_split[left_split]
            right = records_by_split[right_split]
            fields: dict[str, list[str]] = {}
            for field in ("pair_id", "video_id", "caption_id"):
                left_values = {str(getattr(record, field)) for record in left}
                right_values = {str(getattr(record, field)) for record in right}
                shared = sorted(left_values & right_values)
                if shared:
                    fields[field] = shared
            if fields:
                overlaps[f"{left_split}__{right_split}"] = fields
    return overlaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build immutable ELSC manifests from existing assets"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, stage="prepare")
    reports: dict[str, Any] = {"config_hash": config_hash(config), "splits": {}}
    failed = False
    records_by_split: dict[str, list[ManifestRecord]] = {}
    for split in args.splits:
        records, report = prepare_split(config, split)
        records_by_split[split] = records
        reports["splits"][split] = report
        if report["missing_annotation_ids"] or report["missing_feature_count"]:
            failed = True
    overlaps = cross_split_overlaps(records_by_split)
    reports["cross_split_overlaps"] = overlaps
    failed |= bool(overlaps)
    if not overlaps:
        for split, records in records_by_split.items():
            report = reports["splits"][split]
            if report["missing_annotation_ids"] or report["missing_feature_count"]:
                continue
            manifest_path = config["data"][f"{split}_manifest"]
            report["manifest"] = write_manifest(records, manifest_path)
    reports["status"] = "blocked_assets" if failed else "ready"
    output = Path(
        config.get("artifacts", {}).get("prepare_report", "artifacts/manifests/prepare_report.json")
    )
    atomic_json_dump(reports, output)
    meta_output = Path(
        config.get("artifacts", {}).get("manifest_meta", "artifacts/manifests/manifest_meta.json")
    )
    atomic_json_dump(reports, meta_output)
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
