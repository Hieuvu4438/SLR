from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from ocem.data.pseudoclips import (
    PseudoClipError,
    decode_pseudoclip,
    load_pseudoclip_index,
    temporal_sample_start,
    validate_pseudoclip_loader,
)
from ocem.provenance.hashes import canonical_json_sha256, sha256_file


def _pseudo_id(sample_id: str, class_index: int, anchor: int) -> str:
    identity = {
        "source_sample_id": sample_id,
        "class_index": class_index,
        "anchor_start": anchor,
    }
    return hashlib.sha256(
        ("ocem-p14t-pseudo-v1\0" + canonical_json_sha256(identity)).encode()
    ).hexdigest()


def _write_video(path: Path, frames: int = 24) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (32, 24))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = index * 5
        frame[:, :, 1] = np.arange(32, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _record(video: Path, *, split: str, class_index: int) -> dict:
    sample_id = video.stem
    return {
        "schema_version": "ocem.p14t_pseudolabel.v1",
        "pseudo_id": _pseudo_id(sample_id, class_index, 2),
        "source_sample_id": sample_id,
        "source_video": str(video),
        "adaptation_split": split,
        "class_index": class_index,
        "class_name": str(class_index),
        "selection_confidence": 0.9,
        "anchor_start_frame": 2,
        "support_start_frame": 2,
        "support_end_frame_exclusive": 18,
        "upstream_clip_start_frame": 2,
        "upstream_clip_end_frame_exclusive": 17,
        "merged_window_count": 1,
        "merged_window_starts": [2],
        "materialization_semantics": "cico_epoch_pseudo_python_slice_v1",
    }


def _fixture(tmp_path: Path) -> tuple[Path, str, list[dict]]:
    train_video = tmp_path / "train-a.avi"
    holdout_video = tmp_path / "holdout-a.avi"
    _write_video(train_video)
    _write_video(holdout_video)
    records = [
        _record(train_video, split="train", class_index=3),
        _record(holdout_video, split="holdout", class_index=4),
    ]
    index = tmp_path / "index.jsonl"
    index.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return index, sha256_file(index), records


def test_index_filtering_and_content_identity(tmp_path) -> None:
    index, digest, records = _fixture(tmp_path)
    train = load_pseudoclip_index(index, expected_sha256=digest, adaptation_split="train")
    assert train == [records[0]]
    tampered = json.loads(index.read_text().splitlines()[0])
    tampered["class_index"] = 9
    index.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(PseudoClipError, match="pseudo_id content digest mismatch"):
        load_pseudoclip_index(index, expected_sha256=sha256_file(index), adaptation_split="train")


def test_temporal_policy_and_padding_match_pseudo_clip_semantics(tmp_path) -> None:
    _, _, records = _fixture(tmp_path)
    record = records[0]
    assert temporal_sample_start(record, training=False) == 2
    assert temporal_sample_start(record, training=True, rng=random.Random(0)) == 2
    decoded = decode_pseudoclip(record, training=False)
    assert decoded["rgb"].shape == (3, 16, 224, 224)
    assert decoded["observed_frame_count"] == 15
    assert decoded["padded_frame_count"] == 1
    assert np.array_equal(decoded["rgb"][:, -1].numpy(), decoded["rgb"][:, -2].numpy())
    assert decoded["codec_roundtrip"] is False


def test_real_decode_path_smoke_compares_seek_and_sequential(tmp_path) -> None:
    index, digest, _ = _fixture(tmp_path)
    report = validate_pseudoclip_loader(
        index=index,
        expected_index_sha256=digest,
        adaptation_split="train",
        samples=1,
    )
    assert report["status"] == "PASS"
    assert report["batch_shape"] == [1, 3, 16, 224, 224]
    assert report["seek_equals_sequential_for_all"] is True
    assert report["intentional_difference"] == "lossy_mp4v_pseudo-video_roundtrip_removed"
