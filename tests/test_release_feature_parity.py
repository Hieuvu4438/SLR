from __future__ import annotations

import numpy as np

from elsc.features.validate_release import _comparison


def test_release_feature_comparison_requires_shape_and_numerical_match():
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    exact = _comparison(expected.copy(), expected)
    assert exact["passed"] is True
    assert exact["max_abs_error"] == 0.0

    wrong_shape = _comparison(expected[:2], expected)
    assert wrong_shape["passed"] is False

    different = _comparison(expected + 1.0, expected)
    assert different["passed"] is False
