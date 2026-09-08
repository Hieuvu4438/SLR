from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


class RelationError(ValueError):
    """Known positives or excluded negatives violate retrieval semantics."""


@dataclass(frozen=True)
class PairRelations:
    positives: Tensor
    candidates: Tensor
    video_rows: dict[str, int]
    text_columns: dict[str, int]


@dataclass(frozen=True)
class ExcludedNegativeRecord:
    schema_version: str
    video_id: str
    text_id: str
    reason: str
    provenance: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExcludedNegativeRecord":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        missing = sorted(allowed - set(raw))
        if unknown:
            raise RelationError("unknown excluded-negative fields: " + ", ".join(unknown))
        if missing:
            raise RelationError("missing excluded-negative fields: " + ", ".join(missing))
        try:
            record = cls(**dict(raw))
        except TypeError as exc:
            raise RelationError("invalid excluded-negative record") from exc
        if record.schema_version != "excluded_negative.v1":
            raise RelationError("schema_version must be excluded_negative.v1")
        for name in ("video_id", "text_id", "reason", "provenance"):
            if not isinstance(getattr(record, name), str) or not getattr(record, name).strip():
                raise RelationError(f"{name} must be a nonempty string")
        return record


def load_excluded_negatives(
    path: str | Path,
    *,
    video_ids: Iterable[str],
    text_ids: Iterable[str],
    positives_by_video: Mapping[str, Iterable[str]],
) -> tuple[ExcludedNegativeRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise RelationError(f"excluded-negative artifact does not exist: {source}")
    records: list[ExcludedNegativeRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RelationError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise RelationError(f"record at {source}:{line_number} must be an object")
        try:
            records.append(ExcludedNegativeRecord.from_mapping(raw))
        except RelationError as exc:
            raise RelationError(f"{source}:{line_number}: {exc}") from exc
    videos = set(map(str, video_ids))
    texts = set(map(str, text_ids))
    pairs = [(record.video_id, record.text_id) for record in records]
    if len(pairs) != len(set(pairs)):
        raise RelationError("excluded-negative artifact contains duplicate pairs")
    for video_id, text_id in pairs:
        if video_id not in videos or text_id not in texts:
            raise RelationError(f"excluded negative references unknown IDs: {(video_id, text_id)}")
        if text_id in set(map(str, positives_by_video.get(video_id, ()))):
            raise RelationError(f"known positive cannot be an excluded negative: {(video_id, text_id)}")
    return tuple(records)


def relations_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_pair_relations(
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    positives_by_video: Mapping[str, Iterable[str]],
    excluded_pairs: Iterable[tuple[str, str]] = (),
) -> PairRelations:
    videos = tuple(map(str, video_ids))
    texts = tuple(map(str, text_ids))
    if len(set(videos)) != len(videos) or len(set(texts)) != len(texts):
        raise RelationError("batch video/text IDs must be unique")
    video_rows = {item: index for index, item in enumerate(videos)}
    text_columns = {item: index for index, item in enumerate(texts)}
    positives = torch.zeros((len(videos), len(texts)), dtype=torch.bool)
    for video_id, target_ids in positives_by_video.items():
        if video_id not in video_rows:
            continue
        for text_id in target_ids:
            if text_id in text_columns:
                positives[video_rows[video_id], text_columns[text_id]] = True
    candidates = torch.ones_like(positives)
    for video_id, text_id in excluded_pairs:
        if video_id in video_rows and text_id in text_columns:
            candidates[video_rows[video_id], text_columns[text_id]] = False
    if bool((positives & ~candidates).any()):
        raise RelationError("a known positive cannot be excluded from candidates")
    if len(videos) and not bool(positives.any(dim=1).all()):
        raise RelationError("every video query in a training batch needs a known positive")
    if len(texts) and not bool(positives.any(dim=0).all()):
        raise RelationError("every text query in a training batch needs a known positive")
    return PairRelations(positives, candidates, video_rows, text_columns)
