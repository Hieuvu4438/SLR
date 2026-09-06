from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from elsc.data.manifest import ManifestRecord
from elsc.prepare import (
    _validate_temporal_sidecar,
    _validated_feature_sidecar,
    cross_split_overlaps,
    prepare_split,
)
from elsc.utils import sha256_file


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


def test_generated_feature_and_temporal_sidecars_are_hash_checked(tmp_path):
    feature = tmp_path / "video.pkl"
    with feature.open("wb") as handle:
        pickle.dump({"feature": np.zeros((3, 4), dtype=np.float32)}, handle)
    feature_meta = {
        "schema_version": 1,
        "stream_name": "domain_agnostic",
        "source_video_sha256": "source-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "recipe_sha256": "recipe-sha",
        "feature_sha256": sha256_file(feature),
        "feature_shape": [3, 4],
        "feature_dtype": "float32",
    }
    feature.with_suffix(".pkl.meta.json").write_text(
        json.dumps(feature_meta), encoding="utf-8"
    )
    assert _validated_feature_sidecar(feature, (3, 4)) == feature_meta

    temporal = tmp_path / "video.json"
    temporal.write_text(
        json.dumps(
            {
                "verified": True,
                "source_video_sha256": "source-sha",
                "recipe_sha256": "recipe-sha",
                "rf_start": [0, 1, 2],
                "rf_end": [16, 17, 18],
            }
        ),
        encoding="utf-8",
    )
    _validate_temporal_sidecar(
        temporal,
        dense_length=3,
        recipe_sha256="recipe-sha",
        source_video_sha256="source-sha",
    )

    with feature.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validated_feature_sidecar(feature, (3, 4))
