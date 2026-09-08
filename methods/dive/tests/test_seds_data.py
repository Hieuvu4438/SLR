from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import pytest

from dive.adapters import SedsDataError, SedsManifestInputBuilder
from dive.data.manifest import SampleRecord


ROOT = Path(__file__).resolve().parents[3]
SEDS_ROOT = ROOT / "third_party" / "SEDS"
REPRODUCTION_CONFIG = ROOT / "methods" / "dive" / "configs" / "seds_how2sign_reproduction.yaml"


def _record(*, pose_path: str = "train/sample.pkl") -> SampleRecord:
    return SampleRecord(
        schema_version="sample.v1",
        sample_id="sample",
        video_id="sample",
        text_id="sentence",
        split="train",
        sign_language="ase",
        text_language_original="en",
        text_language_model="en",
        text_original="a person signs clearly",
        text_model="a person signs clearly",
        source_video_id="source",
        signer_id=None,
        source_start_sec=0.0,
        source_end_sec=0.8,
        duration_sec=0.8,
        video_path="train/sample.mp4",
        pose_path=pose_path,
        rgb_feature_key="train/sample.pkl",
        translation_artifact_hash=None,
        frame_map_key="frame-map",
        annotation_provenance="fixture",
    )


def _write_native_features(pose_root: Path, rgb_root: Path) -> None:
    pose_path = pose_root / "train" / "sample.pkl"
    rgb_path = rgb_root / "train" / "sample.pkl"
    pose_path.parent.mkdir(parents=True)
    rgb_path.parent.mkdir(parents=True)
    keypoints = np.zeros((20, 133, 3), dtype=np.float32)
    keypoints[:, :, 2] = 1.0
    keypoints[:, :, 0] = np.linspace(20, 220, 133, dtype=np.float32)
    keypoints[:, :, 1] = np.linspace(30, 180, 133, dtype=np.float32)
    pose_payload = {
        "keypoints": keypoints,
        "img_list": [f"frame_{index:04d}.jpg" for index in range(20)],
    }
    with pose_path.open("wb") as handle:
        pickle.dump(pose_payload, handle)
    with rgb_path.open("wb") as handle:
        pickle.dump({"feature": np.ones((5, 1024), dtype=np.float32)}, handle)


def _builder(tmp_path: Path) -> SedsManifestInputBuilder:
    pose_root = tmp_path / "pose"
    rgb_root = tmp_path / "rgb"
    _write_native_features(pose_root, rgb_root)
    return SedsManifestInputBuilder(
        upstream_root=SEDS_ROOT,
        reproduction_config=REPRODUCTION_CONFIG,
        pose_root=pose_root,
        rgb_root=rgb_root,
    )


def test_manifest_builder_preserves_native_shapes_masks_ids_and_rng(tmp_path):
    builder = _builder(tmp_path)
    state = random.getstate()
    video = builder.build_video_batch(
        [_record()], raw_frame_counts=[20], frames_per_second=[25.0]
    )
    assert random.getstate() == state
    assert video.sample_ids == ("sample",)
    assert video.right_pose.shape == (1, 20, 21, 2)
    assert video.left_pose.shape == (1, 20, 21, 2)
    assert video.body_pose.shape == (1, 20, 7, 2)
    assert video.clip_starts.shape == (1, 64)
    assert video.legacy_video_mask.shape == (1, 65)
    assert video.rgb_features.shape == (1, 1024, 64, 1)
    assert int((video.legacy_video_mask == 0).sum()) == 6
    assert video.raw_frame_counts == (20,)
    assert video.frames_per_second == (25.0,)

    text = builder.build_text_batch([_record()])
    assert text.text_ids == ("sentence",)
    assert text.input_ids.shape == (1, 32)
    assert text.token_type_ids.shape == text.attention_mask.shape == (1, 32)
    assert int(text.attention_mask.sum()) > 2


def test_manifest_builder_rejects_duplicate_texts_and_escaped_feature_paths(tmp_path):
    builder = _builder(tmp_path)
    with pytest.raises(SedsDataError, match="unique text IDs"):
        builder.build_text_batch([_record(), _record()])
    with pytest.raises(SedsDataError, match="escapes"):
        builder.build_video_batch([_record(pose_path="../outside.pkl")], raw_frame_counts=[20])
