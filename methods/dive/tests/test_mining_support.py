from __future__ import annotations

import numpy as np
import pytest

from dive.mining.slots import SchemaAudit, validate_strict_numeric_pair
from dive.mining.support import (
    build_endpoint_support,
    differential_support_weights,
    interval_union_length,
    jsd_stability,
    rebin_distribution,
    select_support,
)


def _audit(status: str = "accepted") -> SchemaAudit:
    return SchemaAudit(
        schema_id="strict_numeric_v1",
        audit_status=status,
        artifact_hash="a" * 64 if status == "accepted" else None,
        audit_id="fixture_audit",
    )


def test_strict_numeric_schema_accepts_one_same_slot_measurement_change():
    result = validate_strict_numeric_pair(
        "the value is 12 meters", "the value is 15 meters", _audit()
    )
    assert result.eligible and result.g_sem == 1.0
    assert result.unit_i == result.unit_j == 3
    assert result.category == "strict_numeric_length"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("between 12 and 15 meters", "between 16 and 20 meters"),
        ("about 12 meters", "about 15 meters"),
        ("12 meters or 15 meters", "12 meters or 16 meters"),
        ("if the value is 12 meters", "if the value is 15 meters"),
        ("route 12", "route 15"),
        ("the object is red", "the object is blue"),
    ],
)
def test_strict_numeric_schema_abstains_on_ambiguous_or_nonmeasurement_pairs(first, second):
    assert validate_strict_numeric_pair(first, second, _audit()).eligible is False


def test_pending_schema_never_enters_weighted_bank():
    result = validate_strict_numeric_pair(
        "the value is 12 meters", "the value is 15 meters", _audit("pending")
    )
    assert result.eligible is True
    assert result.g_sem == 0.0
    assert result.reason == "schema_pending"


def test_differential_direction_flips_with_own_and_rival_units():
    root_two = np.sqrt(2.0)
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1 / root_two, 1 / root_two]])
    first = differential_support_weights(features, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.ones(3, bool))
    second = differential_support_weights(features, np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.ones(3, bool))
    np.testing.assert_allclose(first.differential_scores, -second.differential_scores)
    assert first.weights is not None and second.weights is not None
    assert int(first.weights.argmax()) == 0
    assert int(second.weights.argmax()) == 1


def test_absolute_positive_gate_and_tiny_distance_fail_without_forced_support():
    root_two = np.sqrt(2.0)
    features = np.array([[-1.0, 0.0], [-1 / root_two, 1 / root_two]])
    negative = differential_support_weights(
        features, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.ones(2, bool)
    )
    assert negative.reason == "no_positive_mass" and negative.weights is None
    tiny = differential_support_weights(
        np.eye(2), np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.ones(2, bool)
    )
    assert tiny.reason == "tiny_text_distance" and tiny.weights is None


def test_rebin_conserves_mass_with_overlapping_intervals():
    rebinned = rebin_distribution(
        np.array([0.5, 0.5]),
        np.array([[0.0, 3.0], [1.0, 4.0]]),
        np.array([1.0, 3.0]),
        4.0,
    )
    np.testing.assert_allclose(rebinned, np.array([0.5, 0.5]))
    assert rebinned.sum() == pytest.approx(1.0)


def test_jsd_stability_oracles():
    assert jsd_stability(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(1.0)
    assert jsd_stability(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_concentration_uses_floor_cap_and_retained_mass_before_normalization():
    rf = np.array([[index, index + 0.5] for index in range(5)], dtype=np.float64)
    accepted = select_support(np.array([0.30, 0.25, 0.2, 0.15, 0.1]), rf, 5.0)
    assert accepted.reason == "accepted"
    assert accepted.retained_mass == pytest.approx(0.55)
    assert accepted.q is not None and accepted.q.sum() == pytest.approx(1.0)
    diffuse = select_support(np.array([0.26, 0.23, 0.20, 0.16, 0.15]), rf, 5.0)
    assert diffuse.reason == "diffuse_support"
    assert diffuse.retained_mass == pytest.approx(0.49)
    too_short = select_support(np.array([0.5, 0.5]), rf[:2], 2.0)
    assert too_short.reason == "too_few_clips"


def test_rf_union_does_not_double_count_overlap_and_wide_support_fails():
    intervals = np.array([[0.0, 2.0], [1.0, 3.0], [4.0, 5.0]])
    assert interval_union_length(intervals) == pytest.approx(4.0)
    selected = select_support(
        np.array([0.6, 0.2, 0.1, 0.1]),
        np.array([[0.0, 6.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        10.0,
    )
    assert selected.reason == "wide_rf"


def test_identical_view_ids_cannot_claim_two_view_stability():
    result = build_endpoint_support(
        view_features=(np.eye(2), np.eye(2)),
        view_validity=(np.ones(2, bool), np.ones(2, bool)),
        view_rf_intervals=(np.array([[0.0, 1.0], [1.0, 2.0]]),) * 2,
        view_ids=("same", "same"),
        own_unit=np.array([1.0, 0.0]),
        rival_unit=np.array([0.0, 1.0]),
        canonical_centers=np.array([0.5, 1.5]),
        canonical_rf=np.array([[0.0, 1.0], [1.0, 2.0]]),
        duration=2.0,
    )
    assert result.reason == "no_distinct_views" and result.q is None
