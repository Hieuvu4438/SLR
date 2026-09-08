from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping


class ManifestError(ValueError):
    """A manifest violates the DIVE sample contract."""


VALID_SPLITS = frozenset({"train", "dev", "test"})


@dataclass(frozen=True)
class SampleRecord:
    schema_version: str
    sample_id: str
    video_id: str
    text_id: str
    split: str
    sign_language: str
    text_language_original: str
    text_language_model: str
    text_original: str
    text_model: str
    source_video_id: str | None
    signer_id: str | None
    source_start_sec: float | None
    source_end_sec: float | None
    duration_sec: float
    video_path: str | None
    pose_path: str | None
    rgb_feature_key: str | None
    translation_artifact_hash: str | None
    frame_map_key: str | None
    annotation_provenance: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SampleRecord":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        missing = sorted(allowed - set(raw))
        if unknown:
            raise ManifestError("unknown sample fields: " + ", ".join(unknown))
        if missing:
            raise ManifestError("missing sample fields: " + ", ".join(missing))
        record = cls(**dict(raw))
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != "sample.v1":
            raise ManifestError("schema_version must be sample.v1")
        required_text = (
            "sample_id", "video_id", "text_id", "sign_language",
            "text_language_original", "text_language_model", "text_original",
            "text_model", "annotation_provenance",
        )
        for name in required_text:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestError(f"{name} must be a non-empty string")
        if self.split not in VALID_SPLITS:
            raise ManifestError(f"invalid split: {self.split}")
        if isinstance(self.duration_sec, bool) or not isinstance(self.duration_sec, (int, float)):
            raise ManifestError("duration_sec must be numeric")
        if self.duration_sec <= 0:
            raise ManifestError("duration_sec must be positive")
        if (self.source_start_sec is None) != (self.source_end_sec is None):
            raise ManifestError("source start/end must both be null or present")
        if self.source_start_sec is not None:
            assert self.source_end_sec is not None
            if self.source_start_sec < 0 or self.source_end_sec <= self.source_start_sec:
                raise ManifestError("source interval must be nonnegative and increasing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(
    path: str | Path, *, expected_split: str | None = None
) -> tuple[SampleRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise ManifestError(f"manifest does not exist: {source}")
    records: list[SampleRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ManifestError(f"record at {source}:{line_number} must be an object")
        try:
            record = SampleRecord.from_mapping(raw)
        except (ManifestError, TypeError) as exc:
            raise ManifestError(f"{source}:{line_number}: {exc}") from exc
        if expected_split is not None and record.split != expected_split:
            raise ManifestError(
                f"{source}:{line_number}: expected split {expected_split}, got {record.split}"
            )
        records.append(record)
    if not records:
        raise ManifestError(f"manifest is empty: {source}")
    for field in ("sample_id", "video_id"):
        _assert_unique(records, field)
    return tuple(records)


def _assert_unique(records: Iterable[SampleRecord], field: str) -> None:
    seen: set[str] = set()
    repeated: set[str] = set()
    for record in records:
        value = str(getattr(record, field))
        if value in seen:
            repeated.add(value)
        seen.add(value)
    if repeated:
        raise ManifestError(f"duplicate {field}: {sorted(repeated)}")


def validate_split_disjoint(
    split_records: Mapping[str, Iterable[SampleRecord]], *, allow_overlap: bool = False
) -> dict[str, list[str]]:
    records = {name: tuple(items) for name, items in split_records.items()}
    for split, items in records.items():
        if split not in VALID_SPLITS or any(item.split != split for item in items):
            raise ManifestError(f"manifest route does not match split {split}")
    overlaps: dict[str, list[str]] = {}
    split_names = sorted(records)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            for field in ("sample_id", "video_id", "text_id"):
                left = {str(getattr(item, field)) for item in records[first]}
                right = {str(getattr(item, field)) for item in records[second]}
                shared = sorted(left & right)
                if shared:
                    overlaps[f"{first}:{second}:{field}"] = shared
    if overlaps and not allow_overlap:
        raise ManifestError("split overlap: " + ", ".join(sorted(overlaps)))
    return overlaps


def manifest_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_assets(
    records: Iterable[SampleRecord],
    *,
    video_root: str | Path | None,
    pose_root: str | Path | None,
    require_video: bool,
    require_pose: bool,
) -> dict[str, list[str]]:
    video_base = None if video_root is None else Path(video_root)
    pose_base = None if pose_root is None else Path(pose_root)
    missing = {"video": [], "pose": [], "rgb_feature_key": [], "frame_map_key": []}
    for record in records:
        if require_video and (
            video_base is None
            or record.video_path is None
            or not (video_base / record.video_path).is_file()
        ):
            missing["video"].append(record.sample_id)
        if require_pose and (
            pose_base is None
            or record.pose_path is None
            or not (pose_base / record.pose_path).is_file()
        ):
            missing["pose"].append(record.sample_id)
        if record.rgb_feature_key is None:
            missing["rgb_feature_key"].append(record.sample_id)
        if record.frame_map_key is None:
            missing["frame_map_key"].append(record.sample_id)
    return missing
