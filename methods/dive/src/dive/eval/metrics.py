from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DirectionMetrics:
    num_queries: int
    recall: dict[int, float]
    mean_rank: float
    median_rank: float
    ranks: dict[str, int]


@dataclass(frozen=True)
class RetrievalMetrics:
    v2t: DirectionMetrics
    t2v: DirectionMetrics

    def to_dict(self) -> dict[str, object]:
        return {"v2t": asdict(self.v2t), "t2v": asdict(self.t2v)}


def _unique_ids(ids: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(str(item) for item in ids)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} contains duplicate IDs")
    return normalized


def rank_queries(
    scores: np.ndarray,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    relevance: Mapping[str, Iterable[str]],
) -> dict[str, int]:
    """Return one best-positive rank per query using exact score/ID ordering."""
    array = np.asarray(scores)
    queries = _unique_ids(query_ids, "query_ids")
    candidates = _unique_ids(candidate_ids, "candidate_ids")
    if array.shape != (len(queries), len(candidates)):
        raise ValueError(
            f"score shape {array.shape} does not match {(len(queries), len(candidates))}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("full-gallery score matrix must be finite numeric data")
    candidate_set = set(candidates)
    ranks: dict[str, int] = {}
    secondary = np.asarray(candidates, dtype=str)
    for row, query_id in zip(array, queries, strict=True):
        positives = set(map(str, relevance.get(query_id, ())))
        if not positives:
            raise ValueError(f"query has no annotated positives: {query_id}")
        missing = positives - candidate_set
        if missing:
            raise ValueError(f"query {query_id} references missing candidates: {sorted(missing)}")
        # np.lexsort uses the last key as primary: descending exact score, then ID.
        order = np.lexsort((secondary, -row))
        rank_by_index = np.empty(len(order), dtype=np.int64)
        rank_by_index[order] = np.arange(1, len(order) + 1)
        positive_indices = [index for index, candidate_id in enumerate(candidates) if candidate_id in positives]
        ranks[query_id] = int(rank_by_index[positive_indices].min())
    return ranks


def _metrics(ranks: Mapping[str, int], topk: Sequence[int]) -> DirectionMetrics:
    if not ranks:
        raise ValueError("cannot evaluate an empty query set")
    ks = tuple(int(k) for k in topk)
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("topk must contain positive integers")
    values = np.asarray(list(ranks.values()), dtype=np.int64)
    return DirectionMetrics(
        num_queries=len(values),
        recall={k: float(np.mean(values <= k)) for k in ks},
        mean_rank=float(np.mean(values)),
        median_rank=float(np.median(values)),
        ranks=dict(ranks),
    )


def evaluate_retrieval(
    scores_video_text: np.ndarray,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    video_to_text_positives: Mapping[str, Iterable[str]],
    *,
    topk: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Evaluate full-gallery V2T rows and T2V columns without diagonal assumptions."""
    scores = np.asarray(scores_video_text)
    videos = _unique_ids(video_ids, "video_ids")
    texts = _unique_ids(text_ids, "text_ids")
    if scores.shape != (len(videos), len(texts)):
        raise ValueError("scores must have video rows and text columns")
    normalized: dict[str, set[str]] = {
        str(video): set(map(str, targets)) for video, targets in video_to_text_positives.items()
    }
    unknown_video = set(normalized) - set(videos)
    if unknown_video:
        raise ValueError(f"relevance contains unknown video IDs: {sorted(unknown_video)}")
    text_to_video: dict[str, set[str]] = {text: set() for text in texts}
    for video in videos:
        targets = normalized.get(video, set())
        if not targets:
            raise ValueError(f"video query has no annotated positives: {video}")
        unknown_text = targets - set(texts)
        if unknown_text:
            raise ValueError(f"video {video} references unknown text IDs: {sorted(unknown_text)}")
        for text in targets:
            text_to_video[text].add(video)
    for text, targets in text_to_video.items():
        if not targets:
            raise ValueError(f"text query has no annotated positives: {text}")

    v2t_ranks = rank_queries(scores, videos, texts, normalized)
    t2v_ranks = rank_queries(scores.T, texts, videos, text_to_video)
    return RetrievalMetrics(v2t=_metrics(v2t_ranks, topk), t2v=_metrics(t2v_ranks, topk))
