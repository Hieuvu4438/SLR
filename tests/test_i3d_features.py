from __future__ import annotations

import json
import fcntl
from pathlib import Path

import numpy as np
import pytest

from elsc.features.i3d import (
    ExtractionRecipe,
    _acquire_extraction_lock,
    _temporal_metadata,
    preprocess_rgb_frame,
    sliding_window_starts,
)


def test_sliding_windows_match_upstream_tail_alignment():
    assert sliding_window_starts(16) == [0]
    assert sliding_window_starts(20, stride=8) == [0, 4]
    assert sliding_window_starts(33, stride=8) == [0, 8, 16, 17]
    assert sliding_window_starts(4) == [0]


def test_sliding_windows_reject_invalid_counts():
    with pytest.raises(ValueError):
        sliding_window_starts(0)
    with pytest.raises(ValueError):
        sliding_window_starts(16, stride=0)


def test_phoenix_spatial_preprocessing_is_rgb_float_square_resize():
    height, width = 260, 210
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    result = preprocess_rgb_frame(frame)
    assert result.shape == (3, 256, 256)
    assert result.dtype == np.float32
    assert np.allclose(result[0], 1.0, atol=5e-7)
    assert np.allclose(result[1:], 0.0, atol=5e-7)


def test_recipe_digest_changes_with_stride():
    assert ExtractionRecipe(stride=1).digest != ExtractionRecipe(stride=2).digest


def test_temporal_metadata_is_json_round_trip_idempotent():
    recipe = ExtractionRecipe()
    metadata = _temporal_metadata(
        Path("video.mp4"),
        "source-sha",
        17,
        25.0,
        [0, 1],
        recipe,
    )
    assert json.loads(json.dumps(metadata)) == metadata


def test_extraction_lock_times_out_when_same_split_is_owned(tmp_path):
    lock_path = tmp_path / ".extract-train.lock"
    with lock_path.open("w", encoding="utf-8") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with lock_path.open("w", encoding="utf-8") as contender:
            with pytest.raises(RuntimeError, match="timed out"):
                _acquire_extraction_lock(
                    contender,
                    lock_path,
                    wait_seconds=0,
                    poll_seconds=0.01,
                )


def test_extraction_lock_acquires_unowned_split(tmp_path):
    lock_path = tmp_path / ".extract-test.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        waited = _acquire_extraction_lock(
            lock,
            lock_path,
            wait_seconds=0,
            poll_seconds=0.01,
        )
        assert waited >= 0
