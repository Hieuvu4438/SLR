"""Paired-ID full-gallery evaluation with explicit direction and tie policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class EvaluationError(ValueError):
    """Raised when scores cannot represent the locked full gallery."""


@dataclass(frozen=True)
class _Rank:
    primary: int
    optimistic: int
    pessimistic: int
    strict_better: int
    tie_size: int


def _validate_unique(name: str, values: Sequence[str]) -> list[str]:
    normalized = [str(value) for value in values]
    if not normalized or len(set(normalized)) != len(normalized):
        raise EvaluationError(f"{name} must be non-empty and unique")
    return normalized


def _rank(scores: np.ndarray, candidate_ids: Sequence[str], target_id: str) -> _Rank:
    try:
        target_index = candidate_ids.index(target_id)
    except ValueError as error:
        raise EvaluationError(f"target {target_id!r} is absent from the gallery") from error
    target_score = scores[target_index]
    strict_better = int(np.sum(scores > target_score))
    tie_size = int(np.sum(scores == target_score))
    order = sorted(range(len(candidate_ids)), key=lambda index: (-scores[index], candidate_ids[index]))
    primary = order.index(target_index) + 1
    return _Rank(
        primary=primary,
        optimistic=strict_better + 1,
        pessimistic=strict_better + tie_size,
        strict_better=strict_better,
        tie_size=tie_size,
    )


def _recall(ranks: Sequence[float], cutoff: int) -> float:
    return 100.0 * sum(rank <= cutoff for rank in ranks) / len(ranks)


def _summarize(ranks: Sequence[_Rank]) -> dict[str, Any]:
    primary = np.asarray([rank.primary for rank in ranks], dtype=np.float64)
    optimistic = np.asarray([rank.optimistic for rank in ranks], dtype=np.float64)
    pessimistic = np.asarray([rank.pessimistic for rank in ranks], dtype=np.float64)
    expected_rank = np.asarray(
        [rank.strict_better + (rank.tie_size + 1) / 2 for rank in ranks], dtype=np.float64
    )
    result: dict[str, Any] = {
        "queries": len(ranks),
        "R1": _recall(primary, 1),
        "R5": _recall(primary, 5),
        "R10": _recall(primary, 10),
        "median_rank": float(np.median(primary)),
        "mean_rank": float(np.mean(primary)),
        "MRR": float(np.mean(1.0 / primary)),
        "primary_ranks": primary.astype(int).tolist(),
        "tie_queries": int(sum(rank.tie_size > 1 for rank in ranks)),
        "tie_policy": "score_desc_id_asc",
        "tie_sensitivity": {},
    }
    for cutoff in (1, 5, 10):
        expected_hits = [
            float(np.clip((cutoff - rank.strict_better) / rank.tie_size, 0.0, 1.0))
            for rank in ranks
        ]
        result["tie_sensitivity"][f"R{cutoff}"] = {
            "optimistic": _recall(optimistic, cutoff),
            "pessimistic": _recall(pessimistic, cutoff),
            "expected": 100.0 * float(np.mean(expected_hits)),
        }
    result["tie_sensitivity"]["expected_mean_rank"] = float(np.mean(expected_rank))
    result["tie_sensitivity"]["expected_median_rank"] = float(np.median(expected_rank))
    return result


def evaluate_full_gallery(
    scores_t2v: Any,
    scores_v2t: Any,
    *,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    text_to_video: Mapping[str, str],
    video_to_text: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate separate ``[video,text]`` matrices without candidate filtering."""

    videos = _validate_unique("video_ids", video_ids)
    texts = _validate_unique("text_ids", text_ids)
    t2v = np.asarray(scores_t2v)
    v2t = np.asarray(scores_v2t)
    expected_shape = (len(videos), len(texts))
    if t2v.shape != expected_shape or v2t.shape != expected_shape:
        raise EvaluationError(
            f"directional score matrices must both have shape {expected_shape}, "
            f"got {t2v.shape} and {v2t.shape}"
        )
    if not np.issubdtype(t2v.dtype, np.floating) or not np.issubdtype(
        v2t.dtype, np.floating
    ):
        raise EvaluationError("score matrices must have floating dtype")
    if not np.all(np.isfinite(t2v)) or not np.all(np.isfinite(v2t)):
        raise EvaluationError("full-gallery score matrices contain missing or non-finite values")
    if set(text_to_video) != set(texts):
        raise EvaluationError("text_to_video must contain every text query exactly once")
    if set(video_to_text) != set(videos):
        raise EvaluationError("video_to_text must contain every video query exactly once")

    t2v_ranks = [
        _rank(t2v[:, text_index], videos, str(text_to_video[text_id]))
        for text_index, text_id in enumerate(texts)
    ]
    v2t_ranks = [
        _rank(v2t[video_index, :], texts, str(video_to_text[video_id]))
        for video_index, video_id in enumerate(videos)
    ]
    return {
        "schema_version": "ocem.full_gallery_metrics.v1",
        "status": "PASS",
        "matrix_orientation": "[video,text]",
        "gallery": {"videos": len(videos), "texts": len(texts)},
        "coverage": {
            "t2v_scores": int(t2v.size),
            "v2t_scores": int(v2t.size),
            "expected_per_direction": len(videos) * len(texts),
            "fraction": 1.0,
        },
        "T2V": _summarize(t2v_ranks),
        "V2T": _summarize(v2t_ranks),
    }
