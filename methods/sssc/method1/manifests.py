from __future__ import annotations

import csv
import json
import os
import pickle
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

import numpy as np

from .config import Method1Config
from .schemas import GroupRecord, TextRecord, VideoRecord
from .token_spans import canonicalize
from .utils import atomic_json_dump, ordered_hash, sha256_file, sha256_json, sha256_text


class ManifestError(ValueError):
    pass


T = TypeVar("T", TextRecord, VideoRecord, GroupRecord)


def _write_jsonl(records: Iterable[T], path: Path) -> dict[str, Any]:
    materialized = list(records)
    if not materialized:
        raise ManifestError(f"refusing to write empty manifest: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in materialized:
                record.validate()
                handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "schema_version": 1,
        "count": len(materialized),
        "sha256": sha256_file(path),
    }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(f"{source}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ManifestError(f"{source}:{line_number}: record must be an object")
            yield value


def validate_manifest_bundle(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    try:
        metadata = json.loads((root / "manifest_meta.json").read_text(encoding="utf-8"))
        splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
        resources = json.loads((root / "resources.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read completed manifest bundle: {error}") from error
    if metadata.get("status") != "ready" or metadata.get("schema_version") != 1:
        raise ManifestError("manifest metadata is not a completed schema-v1 bundle")
    expected_files = {"texts.jsonl", "videos.jsonl", "groups.jsonl"}
    if set(metadata.get("files", {})) != expected_files:
        raise ManifestError("manifest metadata has an unexpected JSONL file set")
    for name in sorted(expected_files):
        path = root / name
        expected = metadata["files"][name]
        if sha256_file(path) != expected.get("sha256"):
            raise ManifestError(f"manifest hash mismatch: {name}")
        count = sum(1 for _ in iter_jsonl(path))
        if count != expected.get("count"):
            raise ManifestError(f"manifest count mismatch: {name}")
    if sha256_file(root / "splits.json") != metadata.get("splits_sha256"):
        raise ManifestError("split manifest hash mismatch")
    if sha256_file(root / "resources.json") != metadata.get("resources_sha256"):
        raise ManifestError("resource manifest hash mismatch")
    content_sha256 = sha256_json(
        {"files": metadata["files"], "splits": splits, "resources": resources}
    )
    if content_sha256 != metadata.get("content_sha256"):
        raise ManifestError("manifest bundle content hash mismatch")
    return {
        "status": "ready",
        "content_sha256": content_sha256,
        "counts": {
            name: int(metadata["files"][name]["count"])
            for name in sorted(expected_files)
        },
    }


def derive_official_membership(config: Method1Config) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {}
    sources: dict[str, dict[str, Any]] = {}
    if config.data.dataset == "ph":
        if not config.data.official_split_annotations:
            raise ManifestError(
                "official membership file is missing and official_split_annotations were not configured"
            )
        for split in ("train", "dev", "test"):
            path = Path(config.data.official_split_annotations[split])
            if not path.is_file():
                raise ManifestError(f"official {split} annotation is missing: {path}")
            with path.open("r", encoding="utf-8") as handle:
                rows = csv.DictReader(handle, delimiter="|")
                if rows.fieldnames is None or "name" not in rows.fieldnames:
                    raise ManifestError(f"official annotation has no name column: {path}")
                identifiers = [str(row["name"]).strip() for row in rows]
            sources[split] = {"path": str(path), "sha256": sha256_file(path)}
            memberships[split] = identifiers
    else:
        # H2/CSL release pickles already encode the established ordered group membership.
        # This is a mechanical identity extraction, never a random resplit.
        for split in ("train", "dev", "test"):
            path = Path(config.data.source_annotations[split])
            if not path.is_file():
                raise ManifestError(f"source {split} annotation is missing: {path}")
            raw = _trusted_pickle(path)
            if not isinstance(raw, dict):
                raise ManifestError(f"source annotation must be a dictionary: {path}")
            memberships[split] = [str(identifier) for identifier in raw]
            sources[split] = {"path": str(path), "sha256": sha256_file(path)}
    for split, identifiers in memberships.items():
        if not identifiers or any(not identifier for identifier in identifiers):
            raise ManifestError(f"official {split} membership is empty or malformed")
        if len(set(identifiers)) != len(identifiers):
            raise ManifestError(f"official {split} membership contains duplicate IDs")
    overlap = {
        f"{left}_{right}": sorted(set(memberships[left]) & set(memberships[right]))
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(overlap.values()):
        raise ManifestError(f"official PHX splits overlap: {overlap}")
    destination = Path(config.data.official_membership_json)
    payload = {
        "schema_version": 1,
        "dataset": config.data.dataset,
        "derivation": (
            "official_csv_name_column"
            if config.data.dataset == "ph"
            else "ordered_keys_from_established_release_pickles"
        ),
        "source": sources,
        "membership": memberships,
        "ordered_hashes": {split: ordered_hash(values) for split, values in memberships.items()},
    }
    atomic_json_dump(payload, destination)
    return memberships


def load_official_membership(config: Method1Config) -> dict[str, list[str]]:
    path = Path(config.data.official_membership_json)
    if not path.is_file():
        return derive_official_membership(config)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read official membership {path}: {error}") from error
    if isinstance(value, dict) and isinstance(value.get("membership"), dict):
        value = value["membership"]
    if not isinstance(value, dict) or set(value) != {"train", "dev", "test"}:
        raise ManifestError("official membership must contain train/dev/test lists")
    output: dict[str, list[str]] = {}
    for split, raw_ids in value.items():
        if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
            raise ManifestError(f"official {split} membership must be a string list")
        if len(set(raw_ids)) != len(raw_ids):
            raise ManifestError(f"official {split} membership contains duplicate IDs")
        output[split] = raw_ids
    return output


def _trusted_pickle(path: Path) -> Any:
    # Callers must point only to the explicitly configured, audited dataset release.
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted local release is a declared input


def _feature_payload(path: Path) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise ManifestError(f"feature is missing: {path}")
    value = _trusted_pickle(path)
    if isinstance(value, dict):
        value = value.get("feature")
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 1024 or array.shape[0] < 1:
        raise ManifestError(f"feature must have shape [N,1024], got {array.shape}: {path}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ManifestError(f"feature must be numeric and finite: {path}")
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ManifestError(f"invalid feature sidecar: {sidecar}") from error
        if metadata.get("feature_sha256") != digest:
            raise ManifestError(f"feature hash does not match sidecar: {path}")
        if metadata.get("feature_shape") != list(array.shape):
            raise ManifestError(f"feature shape does not match sidecar: {path}")
    return array, digest


def _group_items(raw_group: Any) -> list[dict[str, Any]]:
    if isinstance(raw_group, dict):
        return [raw_group]
    if isinstance(raw_group, list) and raw_group and all(isinstance(item, dict) for item in raw_group):
        return raw_group
    raise ManifestError("annotation group must be a record or a non-empty list of records")


def build_manifests(config: Method1Config) -> dict[str, Any]:
    memberships = load_official_membership(config)
    all_texts: list[TextRecord] = []
    all_videos: list[VideoRecord] = []
    all_groups: list[GroupRecord] = []
    split_records: dict[str, dict[str, list[str]]] = {}
    raw_video_ids_by_split: dict[str, set[str]] = {}
    source_hashes: dict[str, str] = {}

    expected_ph = {"train": 7096, "dev": 519, "test": 642}
    for split in ("train", "dev", "test"):
        source = Path(config.data.source_annotations[split])
        if not source.is_file():
            raise ManifestError(f"source annotation is missing: {source}")
        raw = _trusted_pickle(source)
        if not isinstance(raw, dict):
            raise ManifestError(f"source annotation must be a dictionary: {source}")
        source_hashes[split] = sha256_file(source)
        official_ids = memberships[split]
        missing = [identifier for identifier in official_ids if identifier not in raw]
        if missing:
            raise ManifestError(f"{split} annotation misses {len(missing)} official groups: {missing[:5]}")
        if config.data.dataset == "ph" and len(official_ids) != expected_ph[split]:
            raise ManifestError(
                f"PHX {split} official count must be {expected_ph[split]}, got {len(official_ids)}"
            )
        split_groups: list[str] = []
        split_texts: list[str] = []
        split_videos: list[str] = []
        raw_video_ids: set[str] = set()
        for group_order, raw_group_uid in enumerate(official_ids):
            items = _group_items(raw[raw_group_uid])
            texts = {str(item.get("text", "")).strip() for item in items}
            original_texts = {str(item.get("ori_text", "")).strip() for item in items}
            if len(texts) != 1 or not next(iter(texts)):
                raise ManifestError(f"group {raw_group_uid} has conflicting or empty model text")
            if len(original_texts) > 1:
                raise ManifestError(f"group {raw_group_uid} has conflicting original-language text")
            raw_text = next(iter(texts))
            original_text = next(iter(original_texts)) or None
            group_uid = f"{config.data.dataset}:{split}:group:{raw_group_uid}"
            text_uid = f"{config.data.dataset}:{split}:text:{raw_group_uid}"
            canonical = canonicalize(raw_text)
            text = TextRecord(
                text_uid=text_uid,
                dataset=config.data.dataset,
                split=split,
                group_uid=group_uid,
                official_order=group_order,
                raw_text=raw_text,
                canonical_text=canonical,
                caption_hash=sha256_text(canonical),
                language=config.data.language,
                original_language_text=original_text,
            )
            videos: list[str] = []
            for member_order, item in enumerate(items):
                raw_video_uid = str(item.get("video_name", "")).strip()
                if not raw_video_uid:
                    raise ManifestError(f"group {raw_group_uid} contains a video without video_name")
                if raw_video_uid in raw_video_ids:
                    raise ManifestError(f"duplicate video identity within {split}: {raw_video_uid}")
                raw_video_ids.add(raw_video_uid)
                agnostic_path = Path(config.data.agnostic_root) / split / f"{raw_video_uid}.pkl"
                aware_path = Path(config.data.aware_root) / split / f"{raw_video_uid}.pkl"
                agnostic, agnostic_hash = _feature_payload(agnostic_path)
                aware, aware_hash = _feature_payload(aware_path)
                if agnostic.shape != aware.shape:
                    raise ManifestError(
                        f"feature stream shape mismatch for {raw_video_uid}: {agnostic.shape} vs {aware.shape}"
                    )
                video_uid = f"{config.data.dataset}:{split}:video:{raw_video_uid}"
                video = VideoRecord(
                    video_uid=video_uid,
                    dataset=config.data.dataset,
                    split=split,
                    group_uid=group_uid,
                    official_order=len(split_videos),
                    agnostic_path=str(agnostic_path.resolve()),
                    aware_path=str(aware_path.resolve()),
                    agnostic_sha256=agnostic_hash,
                    aware_sha256=aware_hash,
                    num_feature_rows=int(agnostic.shape[0]),
                    signer_id=None,
                )
                video.validate()
                all_videos.append(video)
                videos.append(video_uid)
                split_videos.append(video_uid)
            text.validate()
            group = GroupRecord(
                group_uid=group_uid,
                video_uids=tuple(videos),
                text_uid=text_uid,
                official_order=group_order,
            )
            group.validate()
            all_texts.append(text)
            all_groups.append(group)
            split_groups.append(group_uid)
            split_texts.append(text_uid)
        raw_video_ids_by_split[split] = raw_video_ids
        split_records[split] = {
            "group_uids": split_groups,
            "text_uids": split_texts,
            "video_uids": split_videos,
            "ordered_group_hash": ordered_hash(split_groups),
            "ordered_text_hash": ordered_hash(split_texts),
            "ordered_video_hash": ordered_hash(split_videos),
        }

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = raw_video_ids_by_split[left] & raw_video_ids_by_split[right]
        if overlap:
            raise ManifestError(f"raw video membership overlaps {left}/{right}: {sorted(overlap)[:5]}")

    destination = Path(config.data.manifest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "texts.jsonl": _write_jsonl(all_texts, destination / "texts.jsonl"),
        "videos.jsonl": _write_jsonl(all_videos, destination / "videos.jsonl"),
        "groups.jsonl": _write_jsonl(all_groups, destination / "groups.jsonl"),
    }
    splits_payload = {"schema_version": 1, "splits": split_records}
    atomic_json_dump(splits_payload, destination / "splits.json")
    resources = {
        "schema_version": 1,
        "dataset": config.data.dataset,
        "release": "configured_audited_release",
        "source_annotation_sha256": source_hashes,
        "official_membership_sha256": sha256_file(config.data.official_membership_json),
        "agnostic_root": str(Path(config.data.agnostic_root).resolve()),
        "aware_root": str(Path(config.data.aware_root).resolve()),
        "agnostic_weight": config.data.agnostic_weight,
        "feature_sampling": config.data.feature_sampling,
        "feature_len": config.data.feature_len,
        "known_unknowns": [
            "contextual feature rows do not provide ground-truth sign boundaries",
            "configured feature provenance must be qualified against the released CiCo regime",
        ],
        "config_sha256": config.digest,
    }
    atomic_json_dump(resources, destination / "resources.json")
    meta = {
        "schema_version": 1,
        "status": "ready",
        "files": files,
        "splits_sha256": sha256_file(destination / "splits.json"),
        "resources_sha256": sha256_file(destination / "resources.json"),
        "content_sha256": sha256_json(
            {
                "files": files,
                "splits": splits_payload,
                "resources": resources,
            }
        ),
    }
    atomic_json_dump(meta, destination / "manifest_meta.json")
    return meta
