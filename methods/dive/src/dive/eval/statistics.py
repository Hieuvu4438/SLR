from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class StatisticsError(ValueError):
    """Paired evaluation inputs do not support the requested analysis."""


@dataclass(frozen=True)
class PairedChanges:
    both_correct: int
    baseline_only: int
    method_only: int
    both_wrong: int


@dataclass(frozen=True)
class BootstrapInterval:
    point_estimate: Mapping[str, float]
    lower: Mapping[str, float]
    upper: Mapping[str, float]
    replicates: int
    seed: int
    resampling_unit: str
    num_units: int


@dataclass(frozen=True)
class SeedAggregate:
    seeds: tuple[int, ...]
    mean: Mapping[str, float]
    sample_std: Mapping[str, float]
    values: Mapping[str, tuple[float, ...]]


def paired_r1_changes(
    baseline_ranks: Mapping[str, int],
    method_ranks: Mapping[str, int],
    *,
    k: int = 1,
) -> PairedChanges:
    if k <= 0 or set(baseline_ranks) != set(method_ranks) or not baseline_ranks:
        raise StatisticsError("paired ranks must be nonempty, aligned, and use positive k")
    outcomes = [(baseline_ranks[key] <= k, method_ranks[key] <= k) for key in baseline_ranks]
    return PairedChanges(
        both_correct=sum(left and right for left, right in outcomes),
        baseline_only=sum(left and not right for left, right in outcomes),
        method_only=sum(not left and right for left, right in outcomes),
        both_wrong=sum(not left and not right for left, right in outcomes),
    )


def _aligned_hits(
    baseline: Mapping[str, int], method: Mapping[str, int], k: int
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if not baseline or set(baseline) != set(method):
        raise StatisticsError("paired bootstrap ranks must be nonempty and exactly aligned")
    ids = tuple(baseline)
    if any(int(baseline[item]) <= 0 or int(method[item]) <= 0 for item in ids):
        raise StatisticsError("ranks must be positive integers")
    return (
        ids,
        np.asarray([baseline[item] <= k for item in ids], dtype=np.float64),
        np.asarray([method[item] <= k for item in ids], dtype=np.float64),
    )


def _sample_query_differences(
    rng: np.random.Generator, baseline: np.ndarray, method: np.ndarray, replicates: int
) -> np.ndarray:
    indices = rng.integers(0, len(baseline), size=(replicates, len(baseline)))
    return (method[indices] - baseline[indices]).mean(axis=1)


def _cluster_indices(ids: Sequence[str], cluster_by_query: Mapping[str, str]) -> dict[str, np.ndarray]:
    if set(ids) != set(cluster_by_query) or any(not str(value) for value in cluster_by_query.values()):
        raise StatisticsError("cluster mapping must exactly cover queries with nonempty unit IDs")
    groups: dict[str, list[int]] = {}
    for index, query_id in enumerate(ids):
        groups.setdefault(str(cluster_by_query[query_id]), []).append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}


def paired_bootstrap_r1(
    baseline_v2t_ranks: Mapping[str, int],
    method_v2t_ranks: Mapping[str, int],
    baseline_t2v_ranks: Mapping[str, int],
    method_t2v_ranks: Mapping[str, int],
    *,
    v2t_cluster_by_query: Mapping[str, str] | None = None,
    t2v_cluster_by_query: Mapping[str, str] | None = None,
    k: int = 1,
    replicates: int = 2000,
    seed: int = 811,
    confidence: float = 0.95,
) -> BootstrapInterval:
    if k <= 0 or replicates < 2 or not 0 < confidence < 1:
        raise StatisticsError("bootstrap k/replicates/confidence are invalid")
    v_ids, v_base, v_method = _aligned_hits(baseline_v2t_ranks, method_v2t_ranks, k)
    t_ids, t_base, t_method = _aligned_hits(baseline_t2v_ranks, method_t2v_ranks, k)
    rng = np.random.default_rng(seed)
    if (v2t_cluster_by_query is None) != (t2v_cluster_by_query is None):
        raise StatisticsError("both directional cluster mappings must be supplied together")
    if v2t_cluster_by_query is None:
        v_replicates = _sample_query_differences(rng, v_base, v_method, replicates)
        t_replicates = _sample_query_differences(rng, t_base, t_method, replicates)
        resampling_unit = "query_level_independent_directions_no_source_ids"
        num_units = len(v_ids) + len(t_ids)
    else:
        assert t2v_cluster_by_query is not None
        v_groups = _cluster_indices(v_ids, v2t_cluster_by_query)
        t_groups = _cluster_indices(t_ids, t2v_cluster_by_query)
        if set(v_groups) != set(t_groups):
            raise StatisticsError(
                "linked source-cluster bootstrap requires the same units in both directions"
            )
        units = tuple(sorted(v_groups))
        v_replicates = np.empty(replicates, dtype=np.float64)
        t_replicates = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            sampled = rng.integers(0, len(units), size=len(units))
            v_indices = np.concatenate([v_groups[units[index]] for index in sampled])
            t_indices = np.concatenate([t_groups[units[index]] for index in sampled])
            v_replicates[replicate] = (v_method[v_indices] - v_base[v_indices]).mean()
            t_replicates[replicate] = (t_method[t_indices] - t_base[t_indices]).mean()
        resampling_unit = "linked_source_cluster_fixed_gallery"
        num_units = len(units)
    endpoint_replicates = 0.5 * (v_replicates + t_replicates)
    point = {
        "v2t_r1_delta": float((v_method - v_base).mean()),
        "t2v_r1_delta": float((t_method - t_base).mean()),
    }
    point["endpoint_delta"] = 0.5 * (point["v2t_r1_delta"] + point["t2v_r1_delta"])
    arrays = {
        "v2t_r1_delta": v_replicates,
        "t2v_r1_delta": t_replicates,
        "endpoint_delta": endpoint_replicates,
    }
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        point_estimate=point,
        lower={key: float(np.quantile(value, tail)) for key, value in arrays.items()},
        upper={key: float(np.quantile(value, 1.0 - tail)) for key, value in arrays.items()},
        replicates=replicates,
        seed=seed,
        resampling_unit=resampling_unit,
        num_units=num_units,
    )


def aggregate_seed_metrics(
    metrics_by_seed: Mapping[int, Mapping[str, float]],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> SeedAggregate:
    if len(metrics_by_seed) < 2:
        raise StatisticsError("seed aggregation requires at least two runs")
    seeds = tuple(sorted(int(seed) for seed in metrics_by_seed))
    if expected_seeds is not None and seeds != tuple(sorted(map(int, expected_seeds))):
        raise StatisticsError("completed seed set differs from the locked experiment plan")
    metric_keys = set(next(iter(metrics_by_seed.values())))
    if not metric_keys or any(set(values) != metric_keys for values in metrics_by_seed.values()):
        raise StatisticsError("all seeds must report the same nonempty metric set")
    values: dict[str, tuple[float, ...]] = {}
    for key in sorted(metric_keys):
        sequence = tuple(float(metrics_by_seed[seed][key]) for seed in seeds)
        if not np.isfinite(sequence).all():
            raise StatisticsError("seed metrics must be finite")
        values[key] = sequence
    return SeedAggregate(
        seeds=seeds,
        mean={key: float(np.mean(sequence)) for key, sequence in values.items()},
        sample_std={key: float(np.std(sequence, ddof=1)) for key, sequence in values.items()},
        values=values,
    )
