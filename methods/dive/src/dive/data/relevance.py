from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence


class RelevanceError(ValueError):
    """A relevance artifact is incomplete or inconsistent with its routed manifest."""


@dataclass(frozen=True)
class RelevanceRecord:
    schema_version: str
    video_id: str
    positive_text_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RelevanceRecord":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        missing = sorted(allowed - set(raw))
        if unknown:
            raise RelevanceError("unknown relevance fields: " + ", ".join(unknown))
        if missing:
            raise RelevanceError("missing relevance fields: " + ", ".join(missing))
        positive_ids = raw.get("positive_text_ids")
        if not isinstance(positive_ids, list):
            raise RelevanceError("positive_text_ids must be a JSON list")
        try:
            record = cls(
                schema_version=raw["schema_version"],
                video_id=raw["video_id"],
                positive_text_ids=tuple(positive_ids),
            )
        except TypeError as exc:
            raise RelevanceError("invalid relevance record") from exc
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != "relevance.v1":
            raise RelevanceError("schema_version must be relevance.v1")
        if not isinstance(self.video_id, str) or not self.video_id:
            raise RelevanceError("video_id must be a nonempty string")
        if not self.positive_text_ids or any(
            not isinstance(item, str) or not item for item in self.positive_text_ids
        ):
            raise RelevanceError("positive_text_ids must contain nonempty strings")
        if len(self.positive_text_ids) != len(set(self.positive_text_ids)):
            raise RelevanceError("positive_text_ids cannot contain duplicates")


def load_relevance(
    path: str | Path,
    *,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
) -> dict[str, frozenset[str]]:
    source = Path(path)
    if not source.is_file():
        raise RelevanceError(f"relevance artifact does not exist: {source}")
    records: list[RelevanceRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RelevanceError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise RelevanceError(f"record at {source}:{line_number} must be an object")
        try:
            records.append(RelevanceRecord.from_mapping(raw))
        except RelevanceError as exc:
            raise RelevanceError(f"{source}:{line_number}: {exc}") from exc
    if not records:
        raise RelevanceError(f"relevance artifact is empty: {source}")
    videos = tuple(map(str, video_ids))
    texts = tuple(map(str, text_ids))
    if len(videos) != len(set(videos)) or len(texts) != len(set(texts)):
        raise RelevanceError("manifest video/text IDs must be unique")
    mapping = {record.video_id: frozenset(record.positive_text_ids) for record in records}
    if len(mapping) != len(records):
        raise RelevanceError("relevance artifact contains duplicate video IDs")
    if set(mapping) != set(videos):
        raise RelevanceError("relevance video IDs do not exactly cover the manifest")
    referenced_texts = set().union(*mapping.values())
    if not referenced_texts <= set(texts):
        raise RelevanceError(
            f"relevance references unknown text IDs: {sorted(referenced_texts - set(texts))}"
        )
    missing_texts = set(texts) - referenced_texts
    if missing_texts:
        raise RelevanceError(f"manifest text IDs have no positive video: {sorted(missing_texts)}")
    return mapping


def relevance_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
