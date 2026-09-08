from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class NeighborError(ValueError):
    """Neighbor mining inputs violate the paired train-set contract."""


@dataclass(frozen=True)
class NeighborProposal:
    pair_id: str
    sample_i: str
    sample_j: str
    nominated_by: tuple[str, ...]
    directions: tuple[str, ...]
    s0_quartet: tuple[tuple[float, float], tuple[float, float]]
    margins0: tuple[float, float, float, float]
    hardness: float


def _normalized(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise NeighborError(f"{name} must be a finite matrix")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-8):
        raise NeighborError(f"{name} contains collapsed rows")
    return array / norms[:, None]


def _margins(quartet: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(quartet[0, 0] - quartet[0, 1]),
        float(quartet[1, 1] - quartet[1, 0]),
        float(quartet[0, 0] - quartet[1, 0]),
        float(quartet[1, 1] - quartet[0, 1]),
    )


def _top_candidates(scores: np.ndarray, eligible: np.ndarray, ids: Sequence[str], limit: int) -> list[int]:
    indices = np.flatnonzero(eligible)
    ordered = sorted(indices.tolist(), key=lambda index: (-float(scores[index]), str(ids[index])))
    return ordered[:limit]


def _pair_id(first: str, second: str) -> str:
    payload = f"neighbor.v1\0{first}\0{second}".encode()
    return "pair_" + hashlib.sha256(payload).hexdigest()[:24]


def mine_neighbor_proposals(
    pooled_video: np.ndarray,
    pooled_text: np.ndarray,
    s0_scores: np.ndarray,
    sample_ids: Sequence[str],
    positives: np.ndarray,
    *,
    shortlist_per_direction: int = 128,
    rerank_topk: int = 16,
) -> tuple[NeighborProposal, ...]:
    """Shortlist with pooled features, then rank hardness only with full S0 quartets."""
    videos = _normalized(pooled_video, "pooled_video")
    texts = _normalized(pooled_text, "pooled_text")
    baseline = np.asarray(s0_scores, dtype=np.float64)
    positive = np.asarray(positives, dtype=np.bool_)
    ids = tuple(map(str, sample_ids))
    count = len(ids)
    if len(set(ids)) != count:
        raise NeighborError("sample IDs must be unique")
    if videos.shape[0] != count or texts.shape[0] != count:
        raise NeighborError("pooled rows must align one-to-one with paired samples")
    if baseline.shape != (count, count) or positive.shape != baseline.shape:
        raise NeighborError("S0/positive matrices must be square and aligned to sample IDs")
    if not np.isfinite(baseline).all():
        raise NeighborError("S0 scores must be finite prelogit values")
    if shortlist_per_direction <= 0 or rerank_topk <= 0:
        raise NeighborError("shortlist and rerank limits must be positive")
    pooled = videos @ texts.T
    nominations: dict[tuple[str, str], dict[str, set[str]]] = {}
    for anchor in range(count):
        row = _top_candidates(
            pooled[anchor], ~positive[anchor], ids, shortlist_per_direction
        )
        column = _top_candidates(
            pooled[:, anchor], ~positive[:, anchor], ids, shortlist_per_direction
        )
        candidates = sorted(set(row) | set(column), key=lambda index: ids[index])
        ranked: list[tuple[float, str, int, tuple[float, ...]]] = []
        for candidate in candidates:
            quartet = baseline[np.ix_([anchor, candidate], [anchor, candidate])]
            margins = _margins(quartet)
            hardness = -min(margins)
            ranked.append((-hardness, ids[candidate], candidate, margins))
        for _, _, candidate, _ in sorted(ranked)[:rerank_topk]:
            first, second = sorted((ids[anchor], ids[candidate]))
            key = (first, second)
            entry = nominations.setdefault(key, {"anchors": set(), "directions": set()})
            entry["anchors"].add(ids[anchor])
            if candidate in row:
                entry["directions"].add("v2t")
            if candidate in column:
                entry["directions"].add("t2v")

    id_to_index = {sample_id: index for index, sample_id in enumerate(ids)}
    output: list[NeighborProposal] = []
    for (first, second), nomination in sorted(nominations.items()):
        i, j = id_to_index[first], id_to_index[second]
        quartet_array = baseline[np.ix_([i, j], [i, j])]
        margins = _margins(quartet_array)
        output.append(
            NeighborProposal(
                pair_id=_pair_id(first, second),
                sample_i=first,
                sample_j=second,
                nominated_by=tuple(sorted(nomination["anchors"])),
                directions=tuple(sorted(nomination["directions"])),
                s0_quartet=(
                    (float(quartet_array[0, 0]), float(quartet_array[0, 1])),
                    (float(quartet_array[1, 0]), float(quartet_array[1, 1])),
                ),
                margins0=margins,
                hardness=-min(margins),
            )
        )
    return tuple(output)


def audit_shortlist_coverage(
    pooled_video: np.ndarray,
    pooled_text: np.ndarray,
    s0_scores: np.ndarray,
    sample_ids: Sequence[str],
    positives: np.ndarray,
    audit_sample_ids: Sequence[str],
    *,
    shortlist_per_direction: int = 128,
    hard_topk: int = 16,
) -> dict[str, object]:
    """Compare pooled shortlist candidates with exact S0-hard neighbors on fixed audit anchors."""
    videos = _normalized(pooled_video, "pooled_video")
    texts = _normalized(pooled_text, "pooled_text")
    baseline = np.asarray(s0_scores, dtype=np.float64)
    positive = np.asarray(positives, dtype=np.bool_)
    ids = tuple(map(str, sample_ids))
    if baseline.shape != (len(ids), len(ids)) or positive.shape != baseline.shape:
        raise NeighborError("coverage audit matrices must align with paired sample IDs")
    id_to_index = {sample_id: index for index, sample_id in enumerate(ids)}
    if len(id_to_index) != len(ids):
        raise NeighborError("coverage audit sample IDs must be unique")
    missing = sorted(set(audit_sample_ids) - set(ids))
    if missing:
        raise NeighborError(f"coverage audit contains unknown anchors: {missing}")
    pooled = videos @ texts.T
    hits = 0
    total = 0
    per_anchor: dict[str, dict[str, object]] = {}
    for sample_id in audit_sample_ids:
        anchor = id_to_index[sample_id]
        row = _top_candidates(pooled[anchor], ~positive[anchor], ids, shortlist_per_direction)
        column = _top_candidates(
            pooled[:, anchor], ~positive[:, anchor], ids, shortlist_per_direction
        )
        approximate = set(row) | set(column)
        exact: list[tuple[float, str, int]] = []
        for candidate in np.flatnonzero(~positive[anchor]).tolist():
            quartet = baseline[np.ix_([anchor, candidate], [anchor, candidate])]
            hardness = -min(_margins(quartet))
            exact.append((-hardness, ids[candidate], candidate))
        exact_top = [candidate for _, _, candidate in sorted(exact)[:hard_topk]]
        anchor_hits = sum(candidate in approximate for candidate in exact_top)
        hits += anchor_hits
        total += len(exact_top)
        per_anchor[sample_id] = {
            "exact_top_ids": [ids[index] for index in exact_top],
            "shortlist_ids": [ids[index] for index in sorted(approximate, key=lambda x: ids[x])],
            "hits": anchor_hits,
            "total": len(exact_top),
        }
    return {
        "schema_version": "shortlist_coverage.v1",
        "query_count": len(audit_sample_ids),
        "hits": hits,
        "total": total,
        "coverage": 1.0 if total == 0 else hits / total,
        "anchors": per_anchor,
    }
