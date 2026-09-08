from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Sequence

import numpy as np
import torch
from torch import Tensor

from .neighbors import NeighborError, NeighborProposal, _margins, _pair_id


PairScorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PooledShortlists:
    video_to_text: np.ndarray
    text_to_video: np.ndarray


@dataclass(frozen=True)
class SparseMiningResult:
    proposals: tuple[NeighborProposal, ...]
    self_scores: np.ndarray
    shortlisted_pair_count: int


def _validated_forbidden(
    forbidden_by_video: Sequence[Collection[int]], count: int
) -> tuple[frozenset[int], ...]:
    if len(forbidden_by_video) != count:
        raise NeighborError("forbidden-candidate rows must align with sample IDs")
    result: list[frozenset[int]] = []
    for row, values in enumerate(forbidden_by_video):
        indices = frozenset(values)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise NeighborError("forbidden-candidate indices must be integers")
        if any(index < 0 or index >= count for index in indices):
            raise NeighborError(f"forbidden-candidate row {row} contains an invalid index")
        result.append(indices | {row})
    return tuple(result)


def _normalized_tensor(values: Tensor, name: str, count: int) -> Tensor:
    if values.ndim != 2 or values.shape[0] != count or not values.is_floating_point():
        raise NeighborError(f"{name} must be a floating matrix aligned to sample IDs")
    floating = values.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(floating).all()):
        raise NeighborError(f"{name} must be finite")
    norms = torch.linalg.vector_norm(floating, dim=1)
    if bool((norms <= 1e-8).any()):
        raise NeighborError(f"{name} contains collapsed rows")
    return floating / norms[:, None]


def _deterministic_topk_rows(
    scores: np.ndarray,
    *,
    id_rank: np.ndarray,
    limit: int,
) -> np.ndarray:
    output = np.empty((scores.shape[0], limit), dtype=np.int64)
    for row_index, row in enumerate(scores):
        finite_count = int(np.isfinite(row).sum())
        if finite_count < limit:
            raise NeighborError(
                f"pooled shortlist row has only {finite_count} eligible candidates, needs {limit}"
            )
        partition = np.argpartition(-row, limit - 1)[:limit]
        threshold = float(row[partition].min())
        strict = np.flatnonzero(row > threshold)
        remaining = limit - len(strict)
        tied = np.flatnonzero(row == threshold)
        tied = tied[np.argsort(id_rank[tied], kind="stable")[:remaining]]
        selected = np.concatenate((strict, tied))
        order = np.lexsort((id_rank[selected], -row[selected]))
        output[row_index] = selected[order]
    return output


def pooled_shortlists_blockwise(
    pooled_video: Tensor,
    pooled_text: Tensor,
    sample_ids: Sequence[str],
    forbidden_by_video: Sequence[Collection[int]],
    *,
    limit: int,
    query_chunk_size: int,
    device: str | torch.device,
) -> PooledShortlists:
    """Compute exact top-K pooled shortlists without allocating a full N-by-N matrix."""
    ids = tuple(map(str, sample_ids))
    count = len(ids)
    if not ids or len(set(ids)) != count:
        raise NeighborError("pooled shortlist sample IDs must be unique and nonempty")
    if limit <= 0 or limit >= count or query_chunk_size <= 0:
        raise NeighborError("pooled shortlist limits/chunk size are invalid")
    forbidden = _validated_forbidden(forbidden_by_video, count)
    inverse: list[set[int]] = [set() for _ in range(count)]
    for video_index, text_indices in enumerate(forbidden):
        for text_index in text_indices:
            inverse[text_index].add(video_index)
    target = torch.device(device)
    videos = _normalized_tensor(pooled_video, "pooled_video", count).to(target)
    texts = _normalized_tensor(pooled_text, "pooled_text", count).to(target)
    id_rank = np.empty(count, dtype=np.int64)
    id_rank[np.asarray(sorted(range(count), key=lambda index: ids[index]))] = np.arange(count)

    def direction(
        queries: Tensor,
        candidates: Tensor,
        blocked: Sequence[Collection[int]],
    ) -> np.ndarray:
        result = np.empty((count, limit), dtype=np.int64)
        for start in range(0, count, query_chunk_size):
            stop = min(count, start + query_chunk_size)
            scores = queries[start:stop] @ candidates.T
            row_indices: list[int] = []
            candidate_indices: list[int] = []
            for local_row, global_row in enumerate(range(start, stop)):
                values = blocked[global_row]
                row_indices.extend([local_row] * len(values))
                candidate_indices.extend(values)
            if row_indices:
                scores[
                    torch.as_tensor(row_indices, dtype=torch.long, device=target),
                    torch.as_tensor(candidate_indices, dtype=torch.long, device=target),
                ] = float("-inf")
            result[start:stop] = _deterministic_topk_rows(
                scores.cpu().numpy(), id_rank=id_rank, limit=limit
            )
        return result

    with torch.no_grad():
        video_to_text = direction(videos, texts, forbidden)
        text_to_video = direction(texts, videos, inverse)
    return PooledShortlists(video_to_text, text_to_video)


def _validate_score_values(values: np.ndarray, expected: int) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (expected,) or not np.isfinite(scores).all():
        raise NeighborError("sparse S0 scorer returned invalid prelogit values")
    if np.any(np.abs(scores) > 1.0 + 1e-5):
        raise NeighborError("sparse S0 scorer escaped cosine bounds")
    return scores


def _shortlist_candidates(
    shortlists: PooledShortlists,
    anchor: int,
    forbidden: Sequence[frozenset[int]],
) -> np.ndarray:
    candidates = np.unique(
        np.concatenate((shortlists.video_to_text[anchor], shortlists.text_to_video[anchor]))
    )
    return np.asarray(
        [
            candidate
            for candidate in candidates.tolist()
            if candidate not in forbidden[anchor] and anchor not in forbidden[candidate]
        ],
        dtype=np.int64,
    )


def rerank_sparse_shortlists(
    shortlists: PooledShortlists,
    sample_ids: Sequence[str],
    forbidden_by_video: Sequence[Collection[int]],
    score_pairs: PairScorer,
    *,
    rerank_topk: int,
) -> SparseMiningResult:
    """Rerank pooled candidates using only exact aligned S0 pair calls."""
    ids = tuple(map(str, sample_ids))
    count = len(ids)
    if not ids or len(set(ids)) != count or rerank_topk <= 0:
        raise NeighborError("sparse reranking IDs/top-K are invalid")
    for name, values in (
        ("video_to_text", shortlists.video_to_text),
        ("text_to_video", shortlists.text_to_video),
    ):
        if values.ndim != 2 or values.shape[0] != count or values.dtype.kind not in "iu":
            raise NeighborError(f"{name} shortlist must be an integer matrix aligned to IDs")
        if np.any(values < 0) or np.any(values >= count):
            raise NeighborError(f"{name} shortlist contains an invalid index")
    forbidden = _validated_forbidden(forbidden_by_video, count)
    candidates_by_anchor = [
        _shortlist_candidates(shortlists, anchor, forbidden) for anchor in range(count)
    ]
    nonempty = [values for values in candidates_by_anchor if len(values)]
    if not nonempty:
        raise NeighborError("pooled shortlists contain no mutually eligible candidate pairs")
    anchors = np.concatenate(
        [
            np.full(len(values), anchor, dtype=np.int64)
            for anchor, values in enumerate(candidates_by_anchor)
            if len(values)
        ]
    )
    candidates = np.concatenate(nonempty)
    lower = np.minimum(anchors, candidates)
    upper = np.maximum(anchors, candidates)
    keys = np.unique(lower * count + upper)
    first = keys // count
    second = keys % count
    self_indices = np.arange(count, dtype=np.int64)
    self_scores = _validate_score_values(score_pairs(self_indices, self_indices), count)
    first_to_second = _validate_score_values(score_pairs(first, second), len(keys))
    second_to_first = _validate_score_values(score_pairs(second, first), len(keys))
    id_rank = {sample_id: rank for rank, sample_id in enumerate(sorted(ids))}
    nominations: dict[int, dict[str, set[str]]] = {}

    for anchor, anchor_candidates in enumerate(candidates_by_anchor):
        if not len(anchor_candidates):
            continue
        anchor_lower = np.minimum(anchor, anchor_candidates)
        anchor_upper = np.maximum(anchor, anchor_candidates)
        positions = np.searchsorted(keys, anchor_lower * count + anchor_upper)
        anchor_is_first = anchor == first[positions]
        cross_out = np.where(
            anchor_is_first, first_to_second[positions], second_to_first[positions]
        )
        cross_in = np.where(anchor_is_first, second_to_first[positions], first_to_second[positions])
        margins = np.stack(
            (
                self_scores[anchor] - cross_out,
                self_scores[anchor_candidates] - cross_in,
                self_scores[anchor] - cross_in,
                self_scores[anchor_candidates] - cross_out,
            ),
            axis=1,
        )
        hardness = -margins.min(axis=1)
        order = sorted(
            range(len(anchor_candidates)),
            key=lambda offset: (
                -float(hardness[offset]),
                id_rank[ids[int(anchor_candidates[offset])]],
            ),
        )[:rerank_topk]
        row_set = set(shortlists.video_to_text[anchor].tolist())
        column_set = set(shortlists.text_to_video[anchor].tolist())
        for offset in order:
            candidate = int(anchor_candidates[offset])
            key = int(min(anchor, candidate) * count + max(anchor, candidate))
            entry = nominations.setdefault(key, {"anchors": set(), "directions": set()})
            entry["anchors"].add(ids[anchor])
            if candidate in row_set:
                entry["directions"].add("v2t")
            if candidate in column_set:
                entry["directions"].add("t2v")

    output: list[NeighborProposal] = []
    for key, nomination in sorted(
        nominations.items(),
        key=lambda item: tuple(sorted((ids[item[0] // count], ids[item[0] % count]))),
    ):
        numeric_first, numeric_second = key // count, key % count
        position = int(np.searchsorted(keys, key))
        if ids[numeric_first] <= ids[numeric_second]:
            i, j = numeric_first, numeric_second
            score_ij = first_to_second[position]
            score_ji = second_to_first[position]
        else:
            i, j = numeric_second, numeric_first
            score_ij = second_to_first[position]
            score_ji = first_to_second[position]
        quartet = np.asarray(
            [[self_scores[i], score_ij], [score_ji, self_scores[j]]], dtype=np.float64
        )
        margins = _margins(quartet)
        output.append(
            NeighborProposal(
                pair_id=_pair_id(ids[i], ids[j]),
                sample_i=ids[i],
                sample_j=ids[j],
                nominated_by=tuple(sorted(nomination["anchors"])),
                directions=tuple(sorted(nomination["directions"])),
                s0_quartet=(
                    (float(quartet[0, 0]), float(quartet[0, 1])),
                    (float(quartet[1, 0]), float(quartet[1, 1])),
                ),
                margins0=margins,
                hardness=-min(margins),
            )
        )
    return SparseMiningResult(tuple(output), self_scores, len(keys))


def audit_sparse_shortlist_coverage(
    shortlists: PooledShortlists,
    sample_ids: Sequence[str],
    forbidden_by_video: Sequence[Collection[int]],
    audit_sample_ids: Sequence[str],
    score_pairs: PairScorer,
    self_scores: np.ndarray,
    *,
    hard_topk: int,
) -> dict[str, object]:
    """Audit pooled recall with full-gallery exact S0 only for fixed train anchors."""
    ids = tuple(map(str, sample_ids))
    count = len(ids)
    id_to_index = {sample_id: index for index, sample_id in enumerate(ids)}
    if len(id_to_index) != count or hard_topk <= 0:
        raise NeighborError("coverage audit IDs/top-K are invalid")
    anchors_requested = tuple(map(str, audit_sample_ids))
    if len(set(anchors_requested)) != len(anchors_requested):
        raise NeighborError("coverage audit anchors must be unique")
    missing = sorted(set(anchors_requested) - set(id_to_index))
    if missing:
        raise NeighborError(f"coverage audit contains unknown anchors: {missing}")
    diagonal = _validate_score_values(self_scores, count)
    forbidden = _validated_forbidden(forbidden_by_video, count)
    hits = 0
    total = 0
    per_anchor: dict[str, dict[str, object]] = {}
    for sample_id in anchors_requested:
        anchor = id_to_index[sample_id]
        candidates = np.asarray(
            [
                candidate
                for candidate in range(count)
                if candidate not in forbidden[anchor] and anchor not in forbidden[candidate]
            ],
            dtype=np.int64,
        )
        if not len(candidates):
            exact_top = np.empty(0, dtype=np.int64)
        else:
            anchors = np.full(len(candidates), anchor, dtype=np.int64)
            cross_out = _validate_score_values(score_pairs(anchors, candidates), len(candidates))
            cross_in = _validate_score_values(score_pairs(candidates, anchors), len(candidates))
            margins = np.stack(
                (
                    diagonal[anchor] - cross_out,
                    diagonal[candidates] - cross_in,
                    diagonal[anchor] - cross_in,
                    diagonal[candidates] - cross_out,
                ),
                axis=1,
            )
            hardness = -margins.min(axis=1)
            order = sorted(
                range(len(candidates)),
                key=lambda offset: (-float(hardness[offset]), ids[int(candidates[offset])]),
            )[:hard_topk]
            exact_top = candidates[np.asarray(order, dtype=np.int64)]
        approximate = set(_shortlist_candidates(shortlists, anchor, forbidden).tolist())
        anchor_hits = sum(int(candidate) in approximate for candidate in exact_top)
        hits += anchor_hits
        total += len(exact_top)
        per_anchor[sample_id] = {
            "exact_top_ids": [ids[int(index)] for index in exact_top],
            "shortlist_ids": [ids[index] for index in sorted(approximate, key=lambda x: ids[x])],
            "hits": anchor_hits,
            "total": len(exact_top),
        }
    return {
        "schema_version": "shortlist_coverage.v1",
        "query_count": len(anchors_requested),
        "hits": hits,
        "total": total,
        "coverage": 1.0 if total == 0 else hits / total,
        "anchors": per_anchor,
    }
