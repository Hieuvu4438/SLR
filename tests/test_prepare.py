from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from elsc.data.manifest import ManifestRecord
from elsc.prepare import cross_split_overlaps, prepare_split


def test_official_ids_filter_merged_dev_annotation(tmp_path: Path):
    annotation = {
        "train-id": {"video_name": "train-video", "ori_text": "de train", "text": "en train"},
        "dev-id": {"video_name": "dev-video", "ori_text": "de dev", "text": "en dev"},
    }
    annotation_path = tmp_path / "dev.pkl"
    with annotation_path.open("wb") as handle:
        pickle.dump(annotation, handle)
    official = tmp_path / "dev.csv"
    official.write_text("name|translation\ndev-id|de dev\n", encoding="utf-8")
    for stream in ("aware", "agnostic"):
        directory = tmp_path / stream / "dev"
        directory.mkdir(parents=True)
        with (directory / "dev-video.pkl").open("wb") as handle:
            pickle.dump({"feature": np.zeros((4, 3), dtype=np.float32)}, handle)
    config = {
        "sources": {
            "dev_annotation": str(annotation_path),
            "dev_official_annotation": str(official),
            "feature_aware_root": str(tmp_path / "aware"),
            "feature_agnostic_root": str(tmp_path / "agnostic"),
            "temporal_metadata_root": None,
        },
        "data": {"dataset": "ph", "feature_dim": 4, "caption_language": "en"},
    }
    records, report = prepare_split(config, "dev")
    assert [record.pair_id for record in records] == ["dev-id"]
    assert records[0].caption_original == "de dev"
    assert records[0].caption_model == "en dev"
    assert report["official_count"] == 1
    assert report["source_extra_count"] == 1
    assert report["missing_feature_count"] == 0


def test_cross_split_overlap_detects_video_leakage():
    def record(split: str, pair: str, video: str) -> ManifestRecord:
        return ManifestRecord(
            1,
            "ph",
            split,
            pair,
            video,
            f"caption-{pair}",
            "text",
            "text",
            "en",
            "agnostic.pkl",
            "aware.pkl",
            4,
            1024,
        )

    overlaps = cross_split_overlaps(
        {
            "train": [record("train", "train-pair", "shared-video")],
            "dev": [record("dev", "dev-pair", "shared-video")],
        }
    )
    assert overlaps == {"dev__train": {"video_id": ["shared-video"]}}
