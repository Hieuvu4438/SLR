from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from slr_common.data.manifest import ManifestRecord, load_manifest
from slr_common.utils import atomic_json_dump, ordered_hash, sha256_file


class GroupIndexError(ValueError):
    pass


@dataclass(frozen=True)
class GroupVideo:
    video_id: str
    original_feature: str
    adapted_feature: str
    dense_length: int
    feature_dim: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GroupVideo":
        try:
            result = cls(**value)
        except TypeError as error:
            raise GroupIndexError(f"invalid video record: {error}") from error
        if not result.video_id or result.dense_length <= 0 or result.feature_dim <= 0:
            raise GroupIndexError("video ID, dense length, and feature dimension must be valid")
        return result


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    canonical_text: str
    original_text: str
    videos: tuple[GroupVideo, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GroupRecord":
        try:
            videos = tuple(GroupVideo.from_dict(item) for item in value["videos"])
            result = cls(
                group_id=str(value["group_id"]),
                canonical_text=str(value["canonical_text"]),
                original_text=str(value.get("original_text", value["canonical_text"])),
                videos=videos,
            )
        except (KeyError, TypeError) as error:
            raise GroupIndexError(f"invalid group record: {error}") from error
        if not result.group_id or not result.canonical_text or not result.videos:
            raise GroupIndexError("every group needs an ID, canonical text, and at least one video")
        ids = [item.video_id for item in result.videos]
        if len(ids) != len(set(ids)):
            raise GroupIndexError(f"duplicate video ID within group {result.group_id}")
        return result


@dataclass(frozen=True)
class CanonicalGroupIndex:
    schema_version: int
    dataset: str
    split: str
    source_path: str
    source_sha256: str
    group_order: str
    groups: tuple[GroupRecord, ...]

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def video_count(self) -> int:
        return sum(len(group.videos) for group in self.groups)

    @property
    def group_id_to_dense(self) -> dict[str, int]:
        return {group.group_id: index for index, group in enumerate(self.groups)}

    def validate(self, *, check_paths: bool = False) -> None:
        if self.schema_version != 1 or not self.dataset or self.split not in {"train", "dev", "test"}:
            raise GroupIndexError("invalid group-index header")
        if self.group_order != "source_insertion_order" or not self.groups:
            raise GroupIndexError("group index must preserve non-empty source insertion order")
        if not Path(self.source_path).is_file():
            raise GroupIndexError(f"index source is unavailable: {self.source_path}")
        if sha256_file(self.source_path) != self.source_sha256:
            raise GroupIndexError("index source SHA-256 no longer matches")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise GroupIndexError("group IDs must be unique")
        all_video_ids: list[str] = []
        for group in self.groups:
            all_video_ids.extend(video.video_id for video in group.videos)
            for video in group.videos:
                original = Path(video.original_feature)
                adapted = Path(video.adapted_feature)
                if original.resolve() == adapted.resolve():
                    raise GroupIndexError(f"feature streams alias for video {video.video_id}")
                if check_paths and (not original.is_file() or not adapted.is_file()):
                    raise GroupIndexError(f"feature stream is unavailable for video {video.video_id}")
        if len(all_video_ids) != len(set(all_video_ids)):
            raise GroupIndexError("video IDs must be unique across groups within a split")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "split": self.split,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "group_order": self.group_order,
            "group_id_to_dense": self.group_id_to_dense,
            "groups": [
                {
                    "group_id": group.group_id,
                    "canonical_text": group.canonical_text,
                    "original_text": group.original_text,
                    "videos": [asdict(video) for video in group.videos],
                }
                for group in self.groups
            ],
        }


def _group_records(records: Iterable[ManifestRecord]) -> tuple[GroupRecord, ...]:
    grouped: OrderedDict[str, list[ManifestRecord]] = OrderedDict()
    for record in records:
        grouped.setdefault(record.caption_id, []).append(record)
    output: list[GroupRecord] = []
    for group_id, members in grouped.items():
        canonical = members[0].caption_model
        original = members[0].caption_original
        mismatched = [
            item.video_id
            for item in members
            if item.caption_model != canonical or item.caption_original != original
        ]
        if mismatched:
            raise GroupIndexError(
                f"group {group_id} has captions inconsistent with its first-record canonical text: "
                + ", ".join(mismatched[:5])
            )
        output.append(
            GroupRecord(
                group_id=group_id,
                canonical_text=canonical,
                original_text=original,
                videos=tuple(
                    GroupVideo(
                        video_id=item.video_id,
                        original_feature=str(Path(item.feature_agnostic).resolve()),
                        adapted_feature=str(Path(item.feature_aware).resolve()),
                        dense_length=item.dense_length,
                        feature_dim=item.feature_dim,
                    )
                    for item in members
                ),
            )
        )
    return tuple(output)


def build_group_index(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    expected_split: str | None = None,
    check_paths: bool = True,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    records = load_manifest(source, expected_split=expected_split)
    index = CanonicalGroupIndex(
        schema_version=1,
        dataset=records[0].dataset,
        split=records[0].split,
        source_path=str(source),
        source_sha256=sha256_file(source),
        group_order="source_insertion_order",
        groups=_group_records(records),
    )
    index.validate(check_paths=check_paths)
    atomic_json_dump(index.to_dict(), output_path)
    return {
        "schema_version": 1,
        "dataset": index.dataset,
        "split": index.split,
        "groups": index.group_count,
        "videos": index.video_count,
        "source_sha256": index.source_sha256,
        "ordered_group_id_hash": ordered_hash(group.group_id for group in index.groups),
        "ordered_video_id_hash": ordered_hash(
            video.video_id for group in index.groups for video in group.videos
        ),
        "index_sha256": sha256_file(output_path),
    }


def load_group_index(path: str | Path, *, check_paths: bool = False) -> CanonicalGroupIndex:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        index = CanonicalGroupIndex(
            schema_version=int(value["schema_version"]),
            dataset=str(value["dataset"]),
            split=str(value["split"]),
            source_path=str(value["source_path"]),
            source_sha256=str(value["source_sha256"]),
            group_order=str(value["group_order"]),
            groups=tuple(GroupRecord.from_dict(item) for item in value["groups"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, GroupIndexError):
            raise
        raise GroupIndexError(f"cannot load group index {source}: {error}") from error
    expected_mapping = index.group_id_to_dense
    if value.get("group_id_to_dense") != expected_mapping:
        raise GroupIndexError("persisted group ID mapping does not match source order")
    index.validate(check_paths=check_paths)
    return index


def assert_split_disjoint(*indexes: CanonicalGroupIndex) -> None:
    seen: dict[str, str] = {}
    for index in indexes:
        for group in index.groups:
            for video in group.videos:
                prior = seen.setdefault(video.video_id, index.split)
                if prior != index.split:
                    raise GroupIndexError(
                        f"video {video.video_id} overlaps splits {prior} and {index.split}"
                    )
