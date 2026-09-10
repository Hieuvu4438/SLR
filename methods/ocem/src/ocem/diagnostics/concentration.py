"""Preregistered support-concentration statistics and paired uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


class ConcentrationError(ValueError):
    """Raised when a diagnostic input cannot support the declared statistic."""


@dataclass(frozen=True)
class ConcentrationStats:
    concentration_p95: float
    excess_kappa: float
    maximum_density: float
    real_mass: float
    null_mass: float
    excess_atom_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return the left-continuous weighted quantile with deterministic tie order."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ConcentrationError("weighted quantile inputs must be aligned nonempty vectors")
    if not 0.0 <= quantile <= 1.0 or not np.all(np.isfinite(values)):
        raise ConcentrationError("weighted quantile values/quantile are invalid")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ConcentrationError("weighted quantile weights must be finite and nonnegative")
    order = np.argsort(values, kind="stable")
    ordered_weights = weights[order] / weights.sum()
    index = int(np.searchsorted(np.cumsum(ordered_weights), quantile, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def independent_concentration(
    affinity: np.ndarray,
    A: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    *,
    epsilon: float = 0.05,
    null_prior: float = 0.15,
    kappa: float = 1.5,
    delta: float = 1e-12,
) -> ConcentrationStats:
    """Measure atom load under the exact closed-form unconstrained reference plan."""

    C = np.asarray(affinity, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if C.ndim != 2 or not C.shape[0] or not C.shape[1]:
        raise ConcentrationError("affinity must have shape [text_tokens, windows]")
    if A.ndim != 2 or A.shape[1] != C.shape[1] or w.shape != (A.shape[0],):
        raise ConcentrationError("A/w do not align with affinity windows")
    if q.shape != (C.shape[1],):
        raise ConcentrationError("q does not align with affinity windows")
    if not all(np.isfinite(value).all() for value in (C, A, w, q)):
        raise ConcentrationError("diagnostic inputs must be finite")
    if epsilon <= 0 or not 0 < null_prior < 1 or kappa <= 0 or delta <= 0:
        raise ConcentrationError("epsilon, null prior, kappa, and delta are invalid")
    if np.any(A < 0) or np.any(w <= 0) or np.any(q <= 0):
        raise ConcentrationError("geometry/reference masses must be positive")
    if not np.allclose(A.sum(axis=0), 1.0, atol=1e-10, rtol=0):
        raise ConcentrationError("A columns must sum to one")
    if not np.isclose(w.sum(), 1.0, atol=1e-10, rtol=0) or not np.isclose(
        q.sum(), 1.0, atol=1e-10, rtol=0
    ):
        raise ConcentrationError("w and q must sum to one")

    real_logits = C / epsilon + np.log((1.0 - null_prior) * q)[None, :]
    logits = np.concatenate(
        (real_logits, np.full((C.shape[0], 1), np.log(null_prior))), axis=1
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    plan = probabilities / C.shape[0]
    real_mass = float(plan[:, :-1].sum())
    null_mass = float(plan[:, -1].sum())
    atom_load = A @ plan[:, :-1].sum(axis=0)
    density = atom_load / (w * max(real_mass, delta))
    excess = np.maximum(atom_load - kappa * w, 0.0)
    return ConcentrationStats(
        concentration_p95=weighted_quantile(density, w, 0.95),
        excess_kappa=float(excess.sum()),
        maximum_density=float(density.max()),
        real_mass=real_mass,
        null_mass=null_mass,
        excess_atom_count=int(np.count_nonzero(excess > 0)),
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    group_ids: Sequence[str] | None = None,
    replicates: int = 10_000,
    seed: int = 20260910,
) -> dict[str, float | int | str]:
    """Paired bootstrap of differences, optionally resampling whole query groups."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ConcentrationError("bootstrap requires at least two finite paired differences")
    if replicates < 1000:
        raise ConcentrationError("bootstrap requires at least 1000 replicates")
    if group_ids is None:
        groups = np.arange(data.size).astype(str)
        mode = "query"
    else:
        if len(group_ids) != data.size:
            raise ConcentrationError("bootstrap group IDs do not align with values")
        groups = np.asarray(list(map(str, group_ids)))
        mode = "cluster"
    unique, inverse = np.unique(groups, return_inverse=True)
    group_sums = np.bincount(inverse, weights=data, minlength=len(unique))
    group_counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    means = group_sums[draws].sum(axis=1) / group_counts[draws].sum(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return {
        "mean": float(data.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "pairs": int(data.size),
        "groups": int(len(unique)),
        "replicates": int(replicates),
        "seed": int(seed),
        "resampling_unit": mode,
    }


def adjusted_paired_effect(
    differences: Sequence[float],
    control_differences: np.ndarray,
    control_names: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = 20260911,
) -> dict[str, object]:
    """Bootstrap the intercept in a paired-difference adjustment model."""

    y = np.asarray(differences, dtype=np.float64)
    X = np.asarray(control_differences, dtype=np.float64)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != y.size:
        raise ConcentrationError("adjustment controls must align with paired differences")
    if len(control_names) != X.shape[1] or y.size < 2:
        raise ConcentrationError("adjustment control names/counts are invalid")
    if not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ConcentrationError("adjustment inputs must be finite")
    means, scales = X.mean(axis=0), X.std(axis=0)
    keep = scales > 1e-12
    names = [str(name) for name, retained in zip(control_names, keep, strict=True) if retained]
    standardized = (X[:, keep] - means[keep]) / scales[keep]
    design = np.column_stack((np.ones(y.size), standardized))

    def intercept(indices: np.ndarray) -> float:
        return float(np.linalg.lstsq(design[indices], y[indices], rcond=None)[0][0])

    observed = intercept(np.arange(y.size))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, y.size, size=y.size)
        estimates[index] = intercept(sample)
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return {
        "adjusted_effect": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "pairs": int(y.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "retained_controls": names,
        "dropped_constant_controls": [
            str(name) for name, retained in zip(control_names, keep, strict=True) if not retained
        ],
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_columns": int(design.shape[1]),
        "model": "paired concentration difference on standardized candidate-control differences",
    }
