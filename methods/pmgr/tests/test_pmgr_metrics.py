from __future__ import annotations

import numpy as np
import pytest

from pmgr.metrics import StreamingGroupMax, evaluate_protocol, group_max_scores


def _fixture():
    scores = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.7, 0.3],
            [0.4, 0.9, 0.2],
            [0.2, 0.0, 0.95],
            [0.1, 0.3, 0.85],
            [0.3, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    owner = [0, 0, 1, 2, 2, 2]
    return scores, owner


def test_numeric_fixture_metrics():
    scores, owner = _fixture()
    result = evaluate_protocol(
        scores,
        video_ids=[f"v{index}" for index in range(6)],
        group_ids=["g0", "g1", "g2"],
        video_to_group=owner,
    )
    assert result["T2V"]["R1"] == 100.0
    assert result["V2T"]["R1"] == pytest.approx(500 / 6)
    assert result["V2T"]["MeanR"] == pytest.approx(4 / 3)
    assert result["V2T"]["MedianR"] == 1.0
    assert result["V2T"]["ranks"][1] == 3


def test_streaming_group_max_equals_dense_across_boundaries():
    scores, owner = _fixture()
    expected = group_max_scores(scores, owner, 3)
    stream = StreamingGroupMax(3)
    stream.update(scores[:2], owner[:2])
    stream.update(scores[2:4], owner[2:4])
    stream.update(scores[4:], owner[4:])
    np.testing.assert_array_equal(stream.finalize(), expected)


def test_stable_tie_policy_uses_candidate_order_not_target():
    scores = np.ones((2, 2))
    result = evaluate_protocol(
        scores,
        video_ids=["v0", "v1"],
        group_ids=["g0", "g1"],
        video_to_group=[0, 1],
    )
    assert result["V2T"]["ranks"] == [1, 2]
    assert result["T2V"]["ranks"] == [1, 2]
    assert result["tie_stats"] == {
        "V2T_queries_with_target_tie": 2,
        "T2V_queries_with_target_tie": 2,
    }
