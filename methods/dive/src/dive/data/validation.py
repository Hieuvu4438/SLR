from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from dive.adapters import SedsDataError, SedsManifestInputBuilder, SedsReproductionError
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


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_prepared_data(
    config: Mapping[str, Any], *, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate all routed splits and register one immutable structural audit."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise DataValidationError("data config must be a mapping")
    resolver = ArtifactResolver(config)
    prepared_parents = []

    def configured_or_prepared(field: str, output_name: str) -> Path:
        value = data.get(field)
        if value is not None:
            return _required_path(value, f"data.{field}")
        try:
            parent = resolver.resolve("prepare_data", output_name, scope="shared")
            prepared_parents.append(parent)
            return parent.path
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
    compact_maps: dict[str, tuple[Any, ...]] = {}
    for split, split_records in records.items():
        frame_path = frame_maps_dir / f"{split}.jsonl"
        maps = load_compact_frame_maps(
            frame_path, expected_sample_ids=[record.sample_id for record in split_records]
        )
        compact_maps[split] = maps
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

    native_input_audit: dict[str, Any] | None = None
    native_frame_maps_dir: Path | None = None
    baseline = config.get("baseline")
    if (
        data.get("dataset") == "how2sign"
        and isinstance(baseline, Mapping)
        and baseline.get("family") == "seds"
    ):
        upstream_root = _required_path(data.get("upstream_root"), "data.upstream_root")
        reproduction_config = _required_path(
            baseline.get("reproduction_config"), "baseline.reproduction_config"
        )
        try:
            builder = SedsManifestInputBuilder(
                upstream_root=upstream_root,
                reproduction_config=reproduction_config,
                pose_root=pose_root,
                rgb_root=rgb_root,
            )
            text = config.get("text")
            tokenizer_artifact = _required_path(
                text.get("tokenizer_artifact") if isinstance(text, Mapping) else None,
                "text.tokenizer_artifact",
            )
            if tokenizer_artifact.resolve() != builder.tokenizer_path:
                raise DataValidationError(
                    "UNVERIFIED_TEXT_MAPPING: configured tokenizer differs from native SEDS"
                )
            native_frame_maps_dir = resolver.output_path("shared", "native_frame_maps")
            split_audits: dict[str, Any] = {}
            for split, split_records in records.items():
                maps_by_id = {item.sample_id: item for item in compact_maps[split]}
                native_rows: list[Mapping[str, Any]] = []
                for start in range(0, len(split_records), 32):
                    part = split_records[start : start + 32]
                    native = builder.build_video_batch(
                        part,
                        raw_frame_counts=[
                            maps_by_id[item.sample_id].video_frame_count for item in part
                        ],
                        frames_per_second=[maps_by_id[item.sample_id].fps for item in part],
                    )
                    if native.pose_raw_frame_indices is None:
                        raise DataValidationError(
                            "NATIVE_SEDS_INPUT_INVALID: selected raw-frame maps are absent"
                        )
                    valid_counts = (native.legacy_video_mask == 0).sum(dim=1) - 1
                    for index, (record, selected) in enumerate(
                        zip(part, native.pose_raw_frame_indices, strict=True)
                    ):
                        valid_count = int(valid_counts[index])
                        starts = native.clip_starts[index, :valid_count].tolist()
                        raw_map = maps_by_id[record.sample_id]
                        native_rows.append(
                            {
                                "schema_version": "seds_native_frame_map.v1",
                                "sample_id": record.sample_id,
                                "video_id": record.video_id,
                                "stored_frame_map_key": record.frame_map_key,
                                "raw_frame_count": raw_map.video_frame_count,
                                "fps": raw_map.fps,
                                "selected_pose_raw_frame_indices": list(selected),
                                "selected_pose_step_count": len(selected),
                                "valid_clip_count": valid_count,
                                "clip_starts_in_selected_pose_steps": starts,
                                "mapping_policy": "native_seds_pose_selected_raw_frames_v1",
                                "rgb_alignment_status": (
                                    "native_count_equal_pose_clips_raw_intervals_pending_asset_audit"
                                ),
                            }
                        )
                native_path = native_frame_maps_dir / f"{split}.jsonl"
                _atomic_jsonl(native_path, native_rows)
                split_audits[split] = {
                    "record_count": len(native_rows),
                    "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                    "selected_pose_step_min": min(
                        row["selected_pose_step_count"] for row in native_rows
                    ),
                    "selected_pose_step_max": max(
                        row["selected_pose_step_count"] for row in native_rows
                    ),
                    "valid_clip_min": min(row["valid_clip_count"] for row in native_rows),
                    "valid_clip_max": max(row["valid_clip_count"] for row in native_rows),
                }
            native_input_audit = {
                "schema_version": "seds_native_input_audit.v1",
                "upstream_root": str(upstream_root.resolve()),
                "reproduction_config": str(reproduction_config.resolve()),
                "tokenizer_artifact": str(tokenizer_artifact.resolve()),
                "native_frame_maps_dir": str(native_frame_maps_dir),
                "splits": split_audits,
                "rgb_pose_clip_counts_equal": True,
                "all_rgb_features_finite": True,
            }
        except (SedsDataError, SedsReproductionError) as exc:
            raise DataValidationError(f"NATIVE_SEDS_INPUT_INVALID: {exc}") from exc

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
        "native_seds_inputs": native_input_audit,
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
    outputs = {"audit": output}
    if native_frame_maps_dir is not None:
        outputs["native_frame_maps_dir"] = native_frame_maps_dir
    resolver.record_stage(
        "validate_data",
        outputs,
        scope="shared",
        parents=tuple(dict.fromkeys(prepared_parents)),
    )
    return report
