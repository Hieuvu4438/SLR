from __future__ import annotations

import numpy as np
import pytest

from dive.eval.metrics import evaluate_retrieval, rank_queries


def test_asymmetric_orientation_has_independent_directional_oracles():
    # Rows: v0,v1; columns: t0,t1,t2. t2 is an additional positive for v0.
    scores = np.array([[0.8, 0.2, 0.7], [0.9, 0.6, 0.1]], dtype=np.float64)
    metrics = evaluate_retrieval(
        scores,
        ["v0", "v1"],
        ["t0", "t1", "t2"],
        {"v0": {"t0", "t2"}, "v1": {"t1"}},
        topk=(1, 2),
    )
    assert metrics.v2t.ranks == {"v0": 1, "v1": 2}
    assert metrics.t2v.ranks == {"t0": 2, "t1": 1, "t2": 1}
    assert metrics.v2t.recall[1] == 0.5
    assert metrics.t2v.recall[1] == pytest.approx(2 / 3)


def test_exact_ties_use_candidate_id_without_extra_queries():
    ranks = rank_queries(
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        ["q0", "q1"],
        ["candidate_b", "candidate_a"],
        {"q0": {"candidate_a"}, "q1": {"candidate_b"}},
    )
    assert ranks == {"q0": 1, "q1": 2}
    assert len(ranks) == 2


def test_multi_positive_uses_best_annotated_rank():
    ranks = rank_queries(
        np.array([[0.2, 0.9, 0.8]]),
        ["q"],
        ["a", "b", "c"],
        {"q": {"a", "c"}},
    )
    assert ranks["q"] == 2


@pytest.mark.parametrize("scores", [np.array([[np.nan]]), np.array([[np.inf]])])
def test_full_gallery_completeness_rejects_nonfinite(scores):
    with pytest.raises(ValueError, match="finite"):
        rank_queries(scores, ["q"], ["c"], {"q": {"c"}})
