from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class EvaluationError(ValueError):
    pass


def stable_positive_ranks(scores: np.ndarray, positive_indexes: Sequence[int]) -> tuple[np.ndarray, int]:
    matrix = np.asarray(scores)
    if matrix.ndim != 2 or matrix.shape[0] != len(positive_indexes):
        raise EvaluationError("scores must be [N_query,N_candidate] with one positive index per query")
    if not np.isfinite(matrix).all():
        raise EvaluationError("score matrix contains NaN or infinity")
    ranks = np.empty(matrix.shape[0], dtype=np.int64)
    ties = 0
    for query, positive in enumerate(positive_indexes):
        if not 0 <= int(positive) < matrix.shape[1]:
            raise EvaluationError(f"positive index out of range for query {query}")
        row = matrix[query]
        positive_score = row[int(positive)]
        ties += int(np.count_nonzero(row == positive_score) > 1)
        order = np.argsort(-row, kind="stable")
        positions = np.flatnonzero(order == int(positive))
        if len(positions) != 1:
            raise AssertionError("stable ordering did not produce exactly one positive rank")
        ranks[query] = int(positions[0]) + 1
    return ranks, ties


def rank_metrics(ranks: Sequence[int]) -> dict[str, float]:
    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1 or not len(values) or bool((values < 1).any()):
        raise EvaluationError("ranks must be a non-empty one-based vector")
    return {
        "R1": float(100.0 * np.mean(values <= 1)),
        "R5": float(100.0 * np.mean(values <= 5)),
        "R10": float(100.0 * np.mean(values <= 10)),
        "MedR": float(np.median(values)),
        "MeanR": float(np.mean(values)),
        "MRR": float(np.mean(1.0 / values)),
    }


def evaluate_grouped_retrieval(
    scores: np.ndarray,
    *,
    video_group_indexes: Sequence[int],
    text_group_indexes: Sequence[int] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Evaluate S[video,text-group] with grouped-max T2V and ordinary V2T."""
    matrix = np.asarray(scores)
    video_groups = np.asarray(video_group_indexes, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != len(video_groups):
        raise EvaluationError("scores must be [N_video,N_text_group]")
    group_count = matrix.shape[1]
    if set(video_groups.tolist()) != set(range(group_count)):
        raise EvaluationError("video_group_indexes must cover each text group contiguously")
    text_groups = (
        np.arange(group_count, dtype=np.int64)
        if text_group_indexes is None
        else np.asarray(text_group_indexes, dtype=np.int64)
    )
    if text_groups.shape != (group_count,) or set(text_groups.tolist()) != set(range(group_count)):
        raise EvaluationError("text_group_indexes must be a permutation of group indexes")

    v2t_ranks, v2t_ties = stable_positive_ranks(matrix, video_groups.tolist())
    grouped = np.full((group_count, group_count), -np.inf, dtype=matrix.dtype)
    # Rows are text queries; columns are candidate video groups.
    for group_index in range(group_count):
        grouped[:, group_index] = matrix[video_groups == group_index].max(axis=0)
    t2v_positive = [int(group) for group in text_groups]
    t2v_ranks, t2v_ties = stable_positive_ranks(grouped, t2v_positive)

    top_k = max(1, min(int(top_k), group_count))
    v2t_order = np.argsort(-matrix, axis=1, kind="stable")[:, :top_k]
    t2v_order = np.argsort(-grouped, axis=1, kind="stable")[:, :top_k]
    return {
        "schema_version": 1,
        "score_orientation": "video_x_text_group",
        "candidate_counts": {"videos": int(matrix.shape[0]), "groups": group_count},
        "tie_policy": "stable_manifest_order",
        "ties": {"V2T": v2t_ties, "T2V": t2v_ties},
        "V2T": {**rank_metrics(v2t_ranks), "ranks": v2t_ranks.tolist()},
        "T2V": {**rank_metrics(t2v_ranks), "ranks": t2v_ranks.tolist()},
        "top_candidate_indexes": {"V2T": v2t_order.tolist(), "T2V": t2v_order.tolist()},
        "grouped_t2v_scores": grouped,
    }
