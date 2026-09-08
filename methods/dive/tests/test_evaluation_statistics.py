from __future__ import annotations

import pytest

from dive.eval.statistics import (
    StatisticsError,
    aggregate_seed_metrics,
    paired_bootstrap_r1,
    paired_r1_changes,
)


def _ranks():
    baseline_v2t = {"v0": 1, "v1": 2, "v2": 3, "v3": 1}
    method_v2t = {"v0": 1, "v1": 1, "v2": 3, "v3": 2}
    baseline_t2v = {"t0": 2, "t1": 2, "t2": 1, "t3": 1}
    method_t2v = {"t0": 1, "t1": 2, "t2": 1, "t3": 1}
    return baseline_v2t, method_v2t, baseline_t2v, method_t2v


def test_paired_changes_count_corrections_and_regressions():
    baseline_v2t, method_v2t, _, _ = _ranks()
    changes = paired_r1_changes(baseline_v2t, method_v2t)
    assert changes.both_correct == 1
    assert changes.baseline_only == 1
    assert changes.method_only == 1
    assert changes.both_wrong == 1


def test_query_level_paired_bootstrap_is_reproducible_and_reports_assumption():
    ranks = _ranks()
    first = paired_bootstrap_r1(*ranks, replicates=200, seed=811)
    second = paired_bootstrap_r1(*ranks, replicates=200, seed=811)
    assert first == second
    assert first.resampling_unit == "query_level_independent_directions_no_source_ids"
    assert first.point_estimate["v2t_r1_delta"] == pytest.approx(0.0)
    assert first.point_estimate["t2v_r1_delta"] == pytest.approx(0.25)
    assert first.point_estimate["endpoint_delta"] == pytest.approx(0.125)


def test_linked_source_cluster_bootstrap_resamples_same_units_for_directions():
    ranks = _ranks()
    v_clusters = {"v0": "s0", "v1": "s0", "v2": "s1", "v3": "s1"}
    t_clusters = {"t0": "s0", "t1": "s0", "t2": "s1", "t3": "s1"}
    result = paired_bootstrap_r1(
        *ranks,
        v2t_cluster_by_query=v_clusters,
        t2v_cluster_by_query=t_clusters,
        replicates=100,
        seed=17,
    )
    assert result.resampling_unit == "linked_source_cluster_fixed_gallery"
    assert result.num_units == 2
    assert result.lower["endpoint_delta"] <= result.point_estimate["endpoint_delta"]
    assert result.upper["endpoint_delta"] >= result.point_estimate["endpoint_delta"]
    with pytest.raises(StatisticsError, match="same units"):
        paired_bootstrap_r1(
            *ranks,
            v2t_cluster_by_query=v_clusters,
            t2v_cluster_by_query={**t_clusters, "t3": "different"},
            replicates=10,
        )


def test_seed_aggregation_reports_mean_sample_std_and_enforces_locked_seeds():
    aggregate = aggregate_seed_metrics(
        {
            17: {"endpoint": 0.60, "r1": 0.55},
            23: {"endpoint": 0.63, "r1": 0.58},
            42: {"endpoint": 0.66, "r1": 0.61},
        },
        expected_seeds=(17, 23, 42),
    )
    assert aggregate.mean["endpoint"] == pytest.approx(0.63)
    assert aggregate.sample_std["endpoint"] == pytest.approx(0.03)
    with pytest.raises(StatisticsError, match="locked experiment plan"):
        aggregate_seed_metrics(
            {17: {"endpoint": 0.6}, 23: {"endpoint": 0.7}},
            expected_seeds=(17, 23, 42),
        )
