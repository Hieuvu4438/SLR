from __future__ import annotations

import numpy as np
import pytest

from ocem.scoring.geometry import GeometryError, build_support_geometry


def test_analytic_overlap_geometry() -> None:
    geometry = build_support_geometry([[0, 2], [1, 3], [8, 10]])
    np.testing.assert_allclose(geometry.atoms, [[0, 1], [1, 2], [2, 3], [8, 10]])
    np.testing.assert_allclose(
        geometry.A,
        [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(geometry.w, [0.2, 0.2, 0.2, 0.4])
    np.testing.assert_allclose(geometry.q, [0.3, 0.3, 0.4])


def test_geometry_is_shift_scale_and_permutation_invariant() -> None:
    intervals = np.array([[0, 2], [1, 3], [8, 10]], dtype=float)
    original = build_support_geometry(intervals)
    transformed = build_support_geometry(intervals * 3 + 17)
    np.testing.assert_allclose(original.A, transformed.A)
    np.testing.assert_allclose(original.w, transformed.w)
    np.testing.assert_allclose(original.q, transformed.q)
    permutation = [2, 0, 1]
    permuted = build_support_geometry(intervals[permutation])
    np.testing.assert_allclose(original.A[:, permutation], permuted.A)
    np.testing.assert_allclose(original.q[permutation], permuted.q)


def test_duplicate_support_splits_reference_mass() -> None:
    geometry = build_support_geometry([[0, 2], [1, 3], [8, 10], [0, 2]])
    assert geometry.q[0] == pytest.approx(geometry.q[3])
    assert geometry.q.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("intervals", [[], [[0, 0]], [[0, float("inf")]]])
def test_invalid_intervals_fail(intervals) -> None:
    with pytest.raises(GeometryError):
        build_support_geometry(intervals)
