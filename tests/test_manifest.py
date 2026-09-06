from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from elsc.data.manifest import ManifestError, ManifestRecord, load_manifest


def _record(pair: str, video: str, caption: str) -> ManifestRecord:
    return ManifestRecord(
        schema_version=1,
        dataset="ph",
        split="dev",
        pair_id=pair,
        video_id=video,
        caption_id=caption,
        caption_original="original",
        caption_model="model text",
        caption_language="en",
        feature_agnostic="agnostic.pkl",
        feature_aware="aware.pkl",
        dense_length=4,
        feature_dim=1024,
    )


def _write(path: Path, records: list[ManifestRecord]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(record)) + "\n" for record in records),
        encoding="utf-8",
    )


def test_duplicate_video_requires_identical_feature_contract(tmp_path: Path):
    first = _record("p0", "v0", "t0")
    second = _record("p1", "v0", "t1")
    second = ManifestRecord(**{**asdict(second), "feature_aware": "different.pkl"})
    path = tmp_path / "manifest.jsonl"
    _write(path, [first, second])
    with pytest.raises(ManifestError, match="inconsistent feature"):
        load_manifest(path)


def test_duplicate_caption_requires_identical_text_contract(tmp_path: Path):
    first = _record("p0", "v0", "t0")
    second = _record("p1", "v1", "t0")
    second = ManifestRecord(**{**asdict(second), "caption_model": "different"})
    path = tmp_path / "manifest.jsonl"
    _write(path, [first, second])
    with pytest.raises(ManifestError, match="inconsistent text"):
        load_manifest(path)
