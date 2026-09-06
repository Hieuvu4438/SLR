from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from elsc.utils import ordered_hash, sha256_file, sha256_json


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestRecord:
    schema_version: int
    dataset: str
    split: str
    pair_id: str
    video_id: str
    caption_id: str
    caption_original: str
    caption_model: str
    caption_language: str
    feature_agnostic: str
    feature_aware: str
    dense_length: int
    feature_dim: int
    temporal_metadata: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManifestRecord":
        try:
            record = cls(**value)
        except TypeError as error:
            raise ManifestError(f"invalid manifest record fields: {error}") from error
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ManifestError("manifest schema_version must be 1")
        if self.split not in {"train", "dev", "test"}:
            raise ManifestError(f"invalid split: {self.split}")
        for name in ("dataset", "pair_id", "video_id", "caption_id", "caption_language"):
            if not getattr(self, name):
                raise ManifestError(f"{name} must be non-empty")
        if self.dense_length <= 0 or self.feature_dim <= 0:
            raise ManifestError("dense_length and feature_dim must be positive")


def iter_manifest(path: str | Path) -> Iterator[ManifestRecord]:
    source = Path(path)
    seen_pairs: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(f"{source}:{line_number}: invalid JSON: {error}") from error
            record = ManifestRecord.from_dict(value)
            if record.pair_id in seen_pairs:
                raise ManifestError(f"duplicate pair_id: {record.pair_id}")
            seen_pairs.add(record.pair_id)
            yield record


def load_manifest(path: str | Path, *, expected_split: str | None = None) -> list[ManifestRecord]:
    records = list(iter_manifest(path))
    if not records:
        raise ManifestError(f"empty manifest: {path}")
    splits = {record.split for record in records}
    if len(splits) != 1:
        raise ManifestError(f"manifest mixes splits: {sorted(splits)}")
    if expected_split is not None and splits != {expected_split}:
        raise ManifestError(f"expected split={expected_split}, found {next(iter(splits))}")
    video_contracts: dict[str, tuple[Any, ...]] = {}
    caption_contracts: dict[str, tuple[Any, ...]] = {}
    for record in records:
        video_contract = (
            record.dataset,
            record.split,
            record.feature_agnostic,
            record.feature_aware,
            record.dense_length,
            record.feature_dim,
            record.temporal_metadata,
        )
        previous_video = video_contracts.setdefault(record.video_id, video_contract)
        if previous_video != video_contract:
            raise ManifestError(
                f"duplicate video_id has inconsistent feature contract: {record.video_id}"
            )
        caption_contract = (
            record.caption_original,
            record.caption_model,
            record.caption_language,
        )
        previous_caption = caption_contracts.setdefault(record.caption_id, caption_contract)
        if previous_caption != caption_contract:
            raise ManifestError(
                f"duplicate caption_id has inconsistent text contract: {record.caption_id}"
            )
    return records


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> dict[str, Any]:
    materialized = list(records)
    if not materialized:
        raise ManifestError("refusing to write an empty manifest")
    pair_ids = [record.pair_id for record in materialized]
    if len(set(pair_ids)) != len(pair_ids):
        raise ManifestError("pair_id values must be unique")
    splits = {record.split for record in materialized}
    if len(splits) != 1:
        raise ManifestError("one manifest file may contain only one split")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in materialized:
            record.validate()
            handle.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")
    return {
        "schema_version": 1,
        "split": next(iter(splits)),
        "count": len(materialized),
        "ordered_pair_id_hash": ordered_hash(pair_ids),
        "ordered_video_id_hash": ordered_hash(record.video_id for record in materialized),
        "ordered_caption_id_hash": ordered_hash(record.caption_id for record in materialized),
        "manifest_sha256": sha256_file(output),
        "records_sha256": sha256_json([asdict(record) for record in materialized]),
    }
