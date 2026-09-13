from __future__ import annotations

import numpy as np

from method1.evaluation import evaluate_grouped_retrieval


def test_hand_calculated_grouped_fixture() -> None:
    scores = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.95]])
    result = evaluate_grouped_retrieval(scores, video_group_indexes=[0, 0, 1])
    assert result["V2T"]["ranks"] == [1, 1, 1]
    assert result["T2V"]["ranks"] == [1, 1]
    np.testing.assert_allclose(result["grouped_t2v_scores"], [[0.9, 0.5], [0.8, 0.95]])


def test_group_max_includes_distractor_and_differs_from_flat_video() -> None:
    scores = np.array([[0.9, 0.8], [0.99, 1.0], [0.5, 0.95]])
    result = evaluate_grouped_retrieval(scores, video_group_indexes=[0, 0, 1])
    assert result["T2V"]["ranks"] == [1, 2]
    assert result["grouped_t2v_scores"][1].tolist() == [1.0, 0.95]


def test_exact_tie_yields_one_stable_rank_per_query() -> None:
    scores = np.array([[1.0, 1.0], [0.0, 2.0]])
    result = evaluate_grouped_retrieval(scores, video_group_indexes=[0, 1])
    assert result["V2T"]["ranks"] == [1, 1]
    assert len(result["V2T"]["ranks"]) == 2
    assert result["ties"]["V2T"] == 1
