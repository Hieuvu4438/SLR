from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import RetrievalMetrics, evaluate_retrieval


class EvaluationProtocolError(ValueError):
    """Evaluation inputs or selection order violate the locked protocol."""


@dataclass(frozen=True)
class GammaCandidate:
    gamma: float
    endpoint: float
    metrics: RetrievalMetrics


@dataclass(frozen=True)
class GammaSelection:
    selected_gamma: float
    candidates: tuple[GammaCandidate, ...]
    no_added_value_on_dev: bool


@dataclass(frozen=True)
class DirectionOpportunity:
    num_queries: int
    baseline_errors: int
    repairable_error_count: Mapping[float, int]
    upper_bound_absolute_r1_gain: Mapping[float, float]
    error_gaps: Mapping[str, float]


@dataclass(frozen=True)
class OpportunityReport:
    v2t: DirectionOpportunity
    t2v: DirectionOpportunity
    interpretation: str = "upper bound on absolute R@1 gain, not predicted accuracy or gain"


@dataclass(frozen=True)
class DuplicateGroup:
    normalized_text: str
    query_ids: tuple[str, ...]
    maximum_top1_hits: int


@dataclass(frozen=True)
class DuplicateCeiling:
    num_queries: int
    groups: tuple[DuplicateGroup, ...]
    maximum_top1_hits: int
    r1_ceiling: float


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def evaluation_ids_hash(
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    video_to_text_positives: Mapping[str, Iterable[str]],
) -> str:
    payload = {
        "video_ids": list(map(str, video_ids)),
        "text_ids": list(map(str, text_ids)),
        "video_to_text_positives": {
            str(video): sorted(map(str, positives))
            for video, positives in sorted(video_to_text_positives.items())
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_gamma_grid(gamma_grid: Sequence[float]) -> tuple[float, ...]:
    gamma_values = tuple(float(item) for item in gamma_grid)
    if not gamma_values or any(not np.isfinite(item) or item < 0 for item in gamma_values):
        raise EvaluationProtocolError("gamma grid must contain finite nonnegative values")
    if len(gamma_values) != len(set(gamma_values)):
        raise EvaluationProtocolError("gamma grid must not contain duplicates")
    if 0.0 not in gamma_values:
        raise EvaluationProtocolError("gamma grid must include the gamma=0 control")
    return tuple(sorted(gamma_values))


def calibrate_locked_checkpoint(
    baseline_scores: np.ndarray,
    correction_scores: np.ndarray,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    video_to_text_positives: Mapping[str, Iterable[str]],
    *,
    gamma_grid: Sequence[float] = (0.0, 0.05, 0.1, 0.2),
    topk: Sequence[int] = (1, 5, 10),
    split: str = "dev",
) -> GammaSelection:
    """Calibrate gamma for one checkpoint that was already selected at gamma_train."""
    if split != "dev":
        raise EvaluationProtocolError("gamma calibration is dev-only")
    baseline = np.asarray(baseline_scores)
    correction = np.asarray(correction_scores)
    if baseline.shape != correction.shape or baseline.ndim != 2:
        raise EvaluationProtocolError("baseline and correction scores must share a 2D layout")
    if not np.isfinite(baseline).all() or not np.isfinite(correction).all():
        raise EvaluationProtocolError("calibration scores must be finite")
    if bool((np.abs(correction) > 1.0 + 1e-6).any()):
        raise EvaluationProtocolError("centered evidence correction must remain in [-1,1]")
    gamma_values = _validate_gamma_grid(gamma_grid)
    candidates: list[GammaCandidate] = []
    for gamma in gamma_values:
        scores = baseline if gamma == 0.0 else baseline + gamma * correction
        if gamma == 0.0 and not np.array_equal(scores, baseline):
            raise EvaluationProtocolError("gamma=0 scores are not bitwise identical to baseline")
        metrics = evaluate_retrieval(
            scores,
            video_ids,
            text_ids,
            video_to_text_positives,
            topk=topk,
        )
        endpoint = 0.5 * (metrics.t2v.recall[1] + metrics.v2t.recall[1])
        candidates.append(GammaCandidate(gamma=gamma, endpoint=endpoint, metrics=metrics))
    winner = min(candidates, key=lambda item: (-item.endpoint, item.gamma))
    return GammaSelection(
        selected_gamma=winner.gamma,
        candidates=tuple(candidates),
        no_added_value_on_dev=winner.gamma == 0.0,
    )


def write_calibration_selection(
    path: str | Path,
    selection: GammaSelection,
    *,
    locked_checkpoint_path: str,
    locked_checkpoint_sha256: str,
    locked_epoch: int,
    locked_optimizer_step: int,
    checkpoint_selection_gamma: float,
    config_hash: str,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    video_to_text_positives: Mapping[str, Iterable[str]],
) -> Path:
    if not locked_checkpoint_path or len(locked_checkpoint_sha256) != 64:
        raise EvaluationProtocolError("locked checkpoint path/SHA-256 provenance is invalid")
    if locked_epoch < 0 or locked_optimizer_step < 0 or checkpoint_selection_gamma <= 0:
        raise EvaluationProtocolError("locked checkpoint selection metadata is invalid")
    if not config_hash:
        raise EvaluationProtocolError("selection artifact requires config hash")
    candidate_payload = []
    for candidate in selection.candidates:
        candidate_payload.append(
            {
                "gamma": candidate.gamma,
                "endpoint": candidate.endpoint,
                "metrics": candidate.metrics.to_dict(),
            }
        )
    payload = {
        "schema_version": "dive_selection.v1",
        "selection_order": "checkpoint_at_fixed_gamma_then_gamma_for_locked_checkpoint",
        "selection_split": "dev",
        "checkpoint": {
            "path": locked_checkpoint_path,
            "sha256": locked_checkpoint_sha256,
            "epoch": locked_epoch,
            "optimizer_step": locked_optimizer_step,
            "selection_gamma": checkpoint_selection_gamma,
        },
        "calibration": {
            "metric": "mean_t2v_v2t_r1",
            "tie_break": "smallest_gamma",
            "selected_gamma": selection.selected_gamma,
            "no_added_value_on_dev": selection.no_added_value_on_dev,
            "candidates": candidate_payload,
        },
        "dev_ids_sha256": evaluation_ids_hash(
            video_ids, text_ids, video_to_text_positives
        ),
        "config_hash": config_hash,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def _normalize_relevance(
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    relevance: Mapping[str, Iterable[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    # Run the exact evaluator as a completeness/ID validation oracle.
    dummy = np.zeros((len(video_ids), len(text_ids)), dtype=np.float64)
    evaluate_retrieval(dummy, video_ids, text_ids, relevance, topk=(1,))
    normalized_input = {
        str(video): set(map(str, positives)) for video, positives in relevance.items()
    }
    v2t = {str(video): normalized_input[str(video)] for video in video_ids}
    t2v = {str(text): set() for text in map(str, text_ids)}
    for video, targets in v2t.items():
        for text in targets:
            t2v[text].add(video)
    return v2t, t2v


def _direction_opportunity(
    scores: np.ndarray,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    relevance: Mapping[str, set[str]],
    gamma_values: Sequence[float],
) -> DirectionOpportunity:
    candidates = np.asarray(tuple(map(str, candidate_ids)), dtype=str)
    errors: dict[str, float] = {}
    for query_index, query_id in enumerate(map(str, query_ids)):
        row = scores[query_index]
        order = np.lexsort((candidates, -row))
        positive_indices = [
            index for index, candidate_id in enumerate(candidates) if candidate_id in relevance[query_id]
        ]
        positions = np.empty(len(order), dtype=np.int64)
        positions[order] = np.arange(len(order))
        best_positive = min(positive_indices, key=lambda index: int(positions[index]))
        top = int(order[0])
        if top in positive_indices:
            continue
        gap = float(row[top] - row[best_positive])
        if gap < -1e-12:
            raise EvaluationProtocolError("opportunity gap is negative under the ranking order")
        errors[query_id] = max(0.0, gap)
    repairable = {
        gamma: sum(gap <= 2.0 * gamma for gap in errors.values()) for gamma in gamma_values
    }
    denominator = len(query_ids)
    return DirectionOpportunity(
        num_queries=denominator,
        baseline_errors=len(errors),
        repairable_error_count=repairable,
        upper_bound_absolute_r1_gain={
            gamma: count / denominator for gamma, count in repairable.items()
        },
        error_gaps=errors,
    )


def opportunity_bound(
    baseline_scores: np.ndarray,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    video_to_text_positives: Mapping[str, Iterable[str]],
    *,
    gamma_grid: Sequence[float] = (0.0, 0.05, 0.1, 0.2),
    split: str = "dev",
) -> OpportunityReport:
    if split != "dev":
        raise EvaluationProtocolError("opportunity analysis is dev-only")
    gamma_values = _validate_gamma_grid(gamma_grid)
    scores = np.asarray(baseline_scores)
    if scores.shape != (len(video_ids), len(text_ids)) or not np.isfinite(scores).all():
        raise EvaluationProtocolError("baseline score matrix is incomplete or nonfinite")
    v2t, t2v = _normalize_relevance(video_ids, text_ids, video_to_text_positives)
    return OpportunityReport(
        v2t=_direction_opportunity(scores, video_ids, text_ids, v2t, gamma_values),
        t2v=_direction_opportunity(scores.T, text_ids, video_ids, t2v, gamma_values),
    )


def duplicate_query_ceiling(
    query_text_by_id: Mapping[str, str],
    relevance: Mapping[str, Iterable[str]],
) -> DuplicateCeiling:
    query_ids = tuple(map(str, query_text_by_id))
    if not query_ids:
        raise EvaluationProtocolError("duplicate analysis requires queries")
    if set(query_ids) != set(map(str, relevance)):
        raise EvaluationProtocolError("duplicate analysis relevance must exactly cover queries")
    grouped: dict[str, list[str]] = {}
    for query_id in query_ids:
        normalized = " ".join(str(query_text_by_id[query_id]).split())
        if not normalized:
            raise EvaluationProtocolError(f"query text is empty after normalization: {query_id}")
        grouped.setdefault(normalized, []).append(query_id)
    maximum_hits = 0
    duplicate_groups: list[DuplicateGroup] = []
    for normalized, group_ids in grouped.items():
        candidate_hits: dict[str, int] = {}
        for query_id in group_ids:
            positives = set(map(str, relevance[query_id]))
            if not positives:
                raise EvaluationProtocolError(f"query has no positives: {query_id}")
            for candidate in positives:
                candidate_hits[candidate] = candidate_hits.get(candidate, 0) + 1
        group_maximum = max(candidate_hits.values())
        maximum_hits += group_maximum
        if len(group_ids) > 1:
            duplicate_groups.append(
                DuplicateGroup(
                    normalized_text=normalized,
                    query_ids=tuple(group_ids),
                    maximum_top1_hits=group_maximum,
                )
            )
    return DuplicateCeiling(
        num_queries=len(query_ids),
        groups=tuple(duplicate_groups),
        maximum_top1_hits=maximum_hits,
        r1_ceiling=maximum_hits / len(query_ids),
    )
