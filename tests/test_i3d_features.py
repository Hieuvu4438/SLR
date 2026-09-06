from __future__ import annotations

import numpy as np
import pytest

from elsc.features.i3d import ExtractionRecipe, preprocess_rgb_frame, sliding_window_starts


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


def test_phoenix_spatial_preprocessing_is_rgb_float_center_crop():
    height, width = 260, 210
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    result = preprocess_rgb_frame(frame)
    assert result.shape == (3, 224, 224)
    assert result.dtype == np.float32
    assert np.all(result[0] == 1.0)
    assert np.all(result[1:] == 0.0)


def test_recipe_digest_changes_with_stride():
    assert ExtractionRecipe(stride=1).digest != ExtractionRecipe(stride=2).digest
