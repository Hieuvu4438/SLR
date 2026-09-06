from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


class EvaluationContractError(ValueError):
    pass


def _direction_metrics(ranks: Sequence[int]) -> dict[str, float | list[int]]:
    array = np.asarray(ranks, dtype=np.int64)
    if not len(array):
        raise EvaluationContractError("no evaluable queries")
    return {
        "R1": float((array < 1).mean() * 100.0),
        "R5": float((array < 5).mean() * 100.0),
        "R10": float((array < 10).mean() * 100.0),
        "MedianR": float(np.median(array) + 1),
        "MeanR": float(np.mean(array) + 1),
        "cols": [int(value) for value in array],
    }


def _ranks(
    scores: np.ndarray,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    positives: Mapping[str, Iterable[str]],
) -> tuple[list[int], list[int], list[dict[str, object]], int, bool]:
    candidate_index = {identifier: index for index, identifier in enumerate(candidate_ids)}
    if len(candidate_index) != len(candidate_ids):
        raise EvaluationContractError("candidate IDs must be unique")
    ranks: list[int] = []
    official_tie_ranks: list[int] = []
    per_query: list[dict[str, object]] = []
    ties = 0
    singleton = True
    for row_index, query_id in enumerate(query_ids):
        positive_ids = list(positives.get(query_id, ()))
        if not positive_ids:
            raise EvaluationContractError(f"query {query_id!r} has no positive mapping")
        singleton &= len(positive_ids) == 1
        try:
            positive_indexes = [candidate_index[item] for item in positive_ids]
        except KeyError as error:
            raise EvaluationContractError(
                f"positive candidate absent from gallery: {error.args[0]}"
            ) from error
        row = scores[row_index]
        best_rank: int | None = None
        best_positive: str | None = None
        query_official_ranks: list[int] = []
        for positive_id, index in zip(positive_ids, positive_indexes, strict=True):
            positive_score = row[index]
            # The official kernel gives every exact tie a matching rank entry. For a
            # per-query scalar we use the optimistic/minimum official rank and report ties.
            rank = int(np.sum(row > positive_score))
            sorted_negative_scores = np.sort(-row)
            tied_ranks = np.flatnonzero(sorted_negative_scores - (-positive_score) == 0).tolist()
            query_official_ranks.extend(int(value) for value in tied_ranks)
            if len(tied_ranks) > 1:
                ties += 1
            if best_rank is None or rank < best_rank:
                best_rank, best_positive = rank, positive_id
        order = np.argsort(-row, kind="stable")
        ranks.append(int(best_rank))
        official_tie_ranks.extend(query_official_ranks)
        per_query.append(
            {
                "query_id": query_id,
                "rank": int(best_rank),
                "matched_positive_id": best_positive,
                "official_tie_ranks": query_official_ranks,
                "ranked_candidate_ids": [candidate_ids[index] for index in order.tolist()],
            }
        )
    return ranks, official_tie_ranks, per_query, ties, singleton


def evaluate_score_matrix(
    scores: np.ndarray,
    *,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    video_to_text: Mapping[str, Iterable[str]],
    text_to_video: Mapping[str, Iterable[str]],
) -> dict[str, object]:
    matrix = np.asarray(scores)
    if matrix.shape != (len(video_ids), len(text_ids)):
        raise EvaluationContractError(
            f"S must be [N_video,N_text], got {matrix.shape} for {(len(video_ids), len(text_ids))}"
        )
    if not np.isfinite(matrix).all():
        raise EvaluationContractError("score matrix contains NaN or infinity")
    v2t_ranks, v2t_official, v2t_queries, v2t_ties, v2t_singleton = _ranks(
        matrix, video_ids, text_ids, video_to_text
    )
    t2v_ranks, t2v_official, t2v_queries, t2v_ties, t2v_singleton = _ranks(
        matrix.T, text_ids, video_ids, text_to_video
    )
    singleton_protocol = v2t_singleton and t2v_singleton
    primary_v2t = v2t_official if singleton_protocol else v2t_ranks
    primary_t2v = t2v_official if singleton_protocol else t2v_ranks
    return {
        "schema_version": 1,
        "score_orientation": "video_x_text",
        "units": "percent",
        "gallery": {"videos": len(video_ids), "texts": len(text_ids)},
        "metric_kernel": (
            "cico_compute_metrics_exact_tie_behavior"
            if singleton_protocol
            else "id_multi_positive_best_rank"
        ),
        "V2T": _direction_metrics(primary_v2t),
        "T2V": _direction_metrics(primary_t2v),
        "diagnostic_best_positive": {
            "V2T": _direction_metrics(v2t_ranks),
            "T2V": _direction_metrics(t2v_ranks),
        },
        "tie_stats": {
            "V2T_queries_with_positive_tie": v2t_ties,
            "T2V_queries_with_positive_tie": t2v_ties,
        },
        "per_query": {"V2T": v2t_queries, "T2V": t2v_queries},
    }


def singleton_positive_maps(
    video_ids: Sequence[str], text_ids: Sequence[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if len(video_ids) != len(text_ids):
        raise EvaluationContractError("singleton protocol requires equal gallery sizes")
    return (
        {video: [text] for video, text in zip(video_ids, text_ids, strict=True)},
        {text: [video] for video, text in zip(video_ids, text_ids, strict=True)},
    )
