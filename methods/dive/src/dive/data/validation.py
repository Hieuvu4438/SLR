from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from dive.artifacts import ArtifactError, ArtifactResolver
from dive.config import config_hash

from .manifest import load_manifest, manifest_hash, validate_assets, validate_split_disjoint
from .relations import load_excluded_negatives, relations_hash
from .relevance import load_relevance, relevance_hash
from .temporal import load_compact_frame_maps


class DataValidationError(ValueError):
    """Prepared data cannot safely enter a model or evaluation stage."""


def _required_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"MISSING_RESOURCE: {field} is not configured")
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.exists():
        raise DataValidationError(f"MISSING_RESOURCE: {field} does not exist: {path}")
    return path


def _hash_strings(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_overlaps(split_records: Mapping[str, tuple[Any, ...]]) -> dict[str, list[str]]:
    sources = {
        split: {record.source_video_id for record in records if record.source_video_id is not None}
        for split, records in split_records.items()
    }
    overlaps: dict[str, list[str]] = {}
    names = sorted(sources)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = sorted(sources[first] & sources[second])
            if shared:
                overlaps[f"{first}:{second}:source_video_id"] = shared
    return overlaps


def validate_prepared_data(
    config: Mapping[str, Any], *, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate all routed splits and register one immutable structural audit."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise DataValidationError("data config must be a mapping")
    resolver = ArtifactResolver(config)

    def configured_or_prepared(field: str, output_name: str) -> Path:
        value = data.get(field)
        if value is not None:
            return _required_path(value, f"data.{field}")
        try:
            return resolver.resolve("prepare_data", output_name, scope="shared").path
        except ArtifactError as exc:
            raise DataValidationError(
                f"MISSING_PARENT_ARTIFACT: data.{field} is null and prepare_data.{output_name} "
                "cannot be resolved"
            ) from exc

    manifests = {
        split: configured_or_prepared(f"{split}_manifest", f"{split}_manifest")
        for split in ("train", "dev", "test")
    }
    records = {
        split: load_manifest(path, expected_split=split) for split, path in manifests.items()
    }
    allow_overlap = bool(data.get("allow_split_overlap", False))
    id_overlaps = validate_split_disjoint(records, allow_overlap=allow_overlap)
    source_overlaps = _source_overlaps(records)
    if source_overlaps and not allow_overlap:
        raise DataValidationError(
            "SPLIT_SOURCE_OVERLAP: " + ", ".join(sorted(source_overlaps))
        )

    video_root = _required_path(data.get("video_root"), "data.video_root")
    pose_root = _required_path(data.get("pose_root"), "data.pose_root")
    rgb_root = _required_path(data.get("rgb_cache_root"), "data.rgb_cache_root")
    missing_assets: dict[str, dict[str, list[str]]] = {}
    for split, split_records in records.items():
        missing = validate_assets(
            split_records,
            video_root=video_root,
            pose_root=pose_root,
            require_video=True,
            require_pose=True,
        )
        missing_assets[split] = missing
        failures = {name: ids for name, ids in missing.items() if ids}
        if failures:
            summary = {name: len(ids) for name, ids in failures.items()}
            raise DataValidationError(f"MISSING_DATA_ASSET: {split}: {summary}")
        missing_rgb = [
            record.sample_id
            for record in split_records
            if record.rgb_feature_key is None
            or not (rgb_root / record.rgb_feature_key).is_file()
        ]
        if missing_rgb:
            raise DataValidationError(
                f"MISSING_RGB_FEATURE: {split}: {len(missing_rgb)} records"
            )

    frame_maps_dir = configured_or_prepared("frame_maps_dir", "frame_maps_dir")
    if not frame_maps_dir.is_dir():
        raise DataValidationError("data.frame_maps_dir must be a directory")
    frame_map_hashes: dict[str, str] = {}
    for split, split_records in records.items():
        frame_path = frame_maps_dir / f"{split}.jsonl"
        maps = load_compact_frame_maps(
            frame_path, expected_sample_ids=[record.sample_id for record in split_records]
        )
        if [item.frame_map_key for item in maps] != [
            record.frame_map_key for record in split_records
        ]:
            raise DataValidationError(
                f"FRAME_MAP_MISMATCH: {split} manifest keys differ from frame-map artifact"
            )
        frame_map_hashes[split] = hashlib.sha256(frame_path.read_bytes()).hexdigest()

    relevance_dir = configured_or_prepared("relevance_dir", "relevance_dir")
    if not relevance_dir.is_dir():
        raise DataValidationError("data.relevance_dir must be a directory")
    relevance_paths = {split: relevance_dir / f"{split}.jsonl" for split in records}
    relevance = {
        split: load_relevance(
            relevance_paths[split],
            video_ids=[record.video_id for record in split_records],
            text_ids=[record.text_id for record in split_records],
        )
        for split, split_records in records.items()
    }
    relations_path = configured_or_prepared("train_relations", "train_relations")
    train_records = records["train"]
    excluded = load_excluded_negatives(
        relations_path,
        video_ids=[record.video_id for record in train_records],
        text_ids=[record.text_id for record in train_records],
        positives_by_video=relevance["train"],
    )

    translation_path: Path | None = None
    translated = [
        record
        for split_records in records.values()
        for record in split_records
        if record.text_language_original != record.text_language_model
    ]
    if translated:
        translation_path = _required_path(
            data.get("translation_artifact"), "data.translation_artifact"
        )
        translation_digest = hashlib.sha256(translation_path.read_bytes()).hexdigest()
        if any(record.translation_artifact_hash != translation_digest for record in translated):
            raise DataValidationError("TRANSLATION_HASH_MISMATCH: manifest translation provenance differs")
    elif any(
        record.translation_artifact_hash is not None
        for split_records in records.values()
        for record in split_records
    ):
        raise DataValidationError("unexpected translation hash when model/original languages match")

    duplicate_text: dict[str, dict[str, Any]] = {}
    for split, split_records in records.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for record in split_records:
            groups[record.text_model].append(record.text_id)
        repeated = sorted(
            (sorted(ids) for ids in groups.values() if len(ids) > 1),
            key=lambda ids: (ids[0], len(ids)),
        )
        duplicate_text[split] = {
            "group_count": len(repeated),
            "text_id_count": sum(map(len, repeated)),
            "ordered_groups_sha256": _hash_strings([item for group in repeated for item in group]),
            "preview": repeated[:10],
        }

    report: dict[str, Any] = {
        "schema_version": "data_validation.v1",
        "ready": True,
        "config_sha256": config_hash(config),
        "dataset": data.get("dataset"),
        "manifests": {
            split: {
                "path": str(path.resolve()),
                "sha256": manifest_hash(path),
                "record_count": len(records[split]),
                "source_video_count": len(
                    {record.source_video_id for record in records[split] if record.source_video_id}
                ),
                "text_languages": dict(
                    sorted(Counter(record.text_language_model for record in records[split]).items())
                ),
            }
            for split, path in manifests.items()
        },
        "relevance": {
            split: {
                "path": str(path.resolve()),
                "sha256": relevance_hash(path),
                "video_query_count": len(relevance[split]),
            }
            for split, path in relevance_paths.items()
        },
        "excluded_negatives": {
            "path": str(relations_path.resolve()),
            "sha256": relations_hash(relations_path),
            "record_count": len(excluded),
        },
        "assets": {
            "video_root": str(video_root.resolve()),
            "pose_root": str(pose_root.resolve()),
            "rgb_cache_root": str(rgb_root.resolve()),
            "missing": missing_assets,
        },
        "frame_maps": {
            "directory": str(frame_maps_dir.resolve()),
            "sha256_by_split": frame_map_hashes,
        },
        "split_id_overlaps": id_overlaps,
        "split_source_overlaps": source_overlaps,
        "duplicate_text_groups": duplicate_text,
        "translated_record_count": len(translated),
        "translation_artifact": None
        if translation_path is None
        else str(translation_path.resolve()),
        "test_content_used_for_tuning": False,
    }
    output = (
        resolver.output_path("shared", "data_validation.json")
        if output_path is None
        else resolver.owned_path("shared", output_path)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    resolver.record_stage("validate_data", {"audit": output}, scope="shared")
    return report
