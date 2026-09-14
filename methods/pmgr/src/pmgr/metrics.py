from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class MetricContractError(ValueError):
    pass


def _metrics(ranks: np.ndarray) -> dict[str, float | list[int]]:
    if ranks.ndim != 1 or not len(ranks):
        raise MetricContractError("rank vector cannot be empty")
    return {
        "R1": float(np.mean(ranks <= 1) * 100.0),
        "R5": float(np.mean(ranks <= 5) * 100.0),
        "R10": float(np.mean(ranks <= 10) * 100.0),
        "MedianR": float(np.median(ranks)),
        "MeanR": float(np.mean(ranks)),
        "ranks": [int(value) for value in ranks],
    }


def _stable_ranks(scores: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, int]:
    if scores.ndim != 2 or targets.shape != (scores.shape[0],):
        raise MetricContractError("targets do not align with score rows")
    if not np.isfinite(scores).all():
        raise MetricContractError("scores contain NaN or infinity")
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise MetricContractError("rank target is outside candidate columns")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    ties = 0
    for row, target in enumerate(targets):
        ranks[row] = int(np.flatnonzero(order[row] == target)[0]) + 1
        ties += int(np.count_nonzero(scores[row] == scores[row, target]) > 1)
    return ranks, ties


def group_max_scores(scores: np.ndarray, video_to_group: Sequence[int], group_count: int) -> np.ndarray:
    matrix = np.asarray(scores)
    owner = np.asarray(video_to_group, dtype=np.int64)
    if matrix.ndim != 2 or owner.shape != (matrix.shape[0],):
        raise MetricContractError("video ownership does not align with the score matrix")
    if matrix.shape[1] != group_count or group_count < 1:
        raise MetricContractError("text candidate count and group count differ")
    output = np.full((group_count, group_count), -np.inf, dtype=matrix.dtype)
    for group in range(group_count):
        members = np.flatnonzero(owner == group)
        if not len(members):
            raise MetricContractError(f"candidate group {group} has no video")
        output[:, group] = matrix[members].max(axis=0)
    return output


def evaluate_protocol(
    scores: np.ndarray,
    *,
    video_ids: Sequence[str],
    group_ids: Sequence[str],
    video_to_group: Sequence[int],
) -> dict[str, object]:
    matrix = np.asarray(scores)
    if matrix.shape != (len(video_ids), len(group_ids)):
        raise MetricContractError("score matrix must be [videos, existing text groups]")
    if len(video_ids) != len(set(video_ids)) or len(group_ids) != len(set(group_ids)):
        raise MetricContractError("video IDs and group IDs must each be unique")
    owner = np.asarray(video_to_group, dtype=np.int64)
    v2t_ranks, v2t_ties = _stable_ranks(matrix, owner)
    grouped = group_max_scores(matrix, owner, len(group_ids))
    targets = np.arange(len(group_ids), dtype=np.int64)
    t2v_ranks, t2v_ties = _stable_ranks(grouped, targets)
    return {
        "schema_version": 1,
        "score_orientation": "video_x_text_group",
        "metric_policy": "stable_candidate_order_v1",
        "gallery": {"videos": len(video_ids), "groups": len(group_ids)},
        "V2T": _metrics(v2t_ranks),
        "T2V": _metrics(t2v_ranks),
        "tie_stats": {"V2T_queries_with_target_tie": v2t_ties, "T2V_queries_with_target_tie": t2v_ties},
    }


@dataclass
class StreamingGroupMax:
    group_count: int

    def __post_init__(self) -> None:
        if self.group_count < 1:
            raise MetricContractError("group_count must be positive")
        self.values: np.ndarray | None = None
        self.seen = np.zeros(self.group_count, dtype=bool)

    def update(self, score_rows: np.ndarray, owner_rows: Sequence[int]) -> None:
        block = np.asarray(score_rows)
        owners = np.asarray(owner_rows, dtype=np.int64)
        if block.ndim != 2 or block.shape[1] != self.group_count or owners.shape != (len(block),):
            raise MetricContractError("streamed score block has an invalid shape")
        if self.values is None:
            self.values = np.full((self.group_count, self.group_count), -np.inf, dtype=block.dtype)
        for row, owner in zip(block, owners, strict=True):
            if owner < 0 or owner >= self.group_count:
                raise MetricContractError("streamed video owner is outside group range")
            self.values[:, owner] = np.maximum(self.values[:, owner], row)
            self.seen[owner] = True

    def finalize(self) -> np.ndarray:
        if self.values is None or not bool(self.seen.all()):
            missing = np.flatnonzero(~self.seen).tolist()
            raise MetricContractError(f"stream omitted complete candidate groups: {missing}")
        return self.values
