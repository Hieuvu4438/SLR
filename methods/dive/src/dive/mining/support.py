from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SupportError(ValueError):
    """Support inputs violate numerical or temporal geometry contracts."""


@dataclass(frozen=True)
class DifferentialWeights:
    weights: np.ndarray | None
    text_distance: float
    positive_scores: np.ndarray
    differential_scores: np.ndarray
    reason: str


@dataclass(frozen=True)
class SupportSelection:
    q: np.ndarray | None
    retained_mass: float
    rf_union_ratio: float | None
    reason: str


@dataclass(frozen=True)
class EndpointSupport:
    q: np.ndarray | None
    text_distance: float
    stability: float | None
    retained_mass: float
    rf_union_ratio: float | None
    reason: str


@dataclass(frozen=True)
class PairSupport:
    endpoint_i: EndpointSupport
    endpoint_j: EndpointSupport
    reliability: float
    reason: str


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise SupportError(f"{name} must be a finite matrix")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-6) or not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-4):
        raise SupportError(f"{name} rows must be normalized and non-collapsed")
    return array


def differential_support_weights(
    reference_features: np.ndarray,
    own_unit: np.ndarray,
    rival_unit: np.ndarray,
    valid: np.ndarray,
    *,
    h_min: float = 1e-3,
    mass_epsilon: float = 1e-12,
    tau_localization: float = 0.07,
    eta_positive: float = 0.0,
    eta_differential: float = 0.05,
) -> DifferentialWeights:
    features = _normalized_rows(reference_features, "reference_features")
    own = _normalized_rows(np.asarray(own_unit)[None, :], "own_unit")[0]
    rival = _normalized_rows(np.asarray(rival_unit)[None, :], "rival_unit")[0]
    mask = np.asarray(valid, dtype=np.bool_)
    if mask.shape != (len(features),):
        raise SupportError("valid mask must have one entry per reference feature")
    if h_min <= 0 or mass_epsilon <= 0 or tau_localization <= 0:
        raise SupportError("h_min, mass_epsilon and tau_localization must be positive")
    text_distance = float(np.linalg.norm(own - rival))
    positive = features @ own
    differential = np.zeros(len(features), dtype=np.float64)
    if text_distance < h_min:
        return DifferentialWeights(None, text_distance, positive, differential, "tiny_text_distance")
    if not mask.any():
        return DifferentialWeights(None, text_distance, positive, differential, "no_valid_clips")
    differential = features @ ((own - rival) / text_distance)
    logits = positive[mask] / tau_localization
    logits = logits - logits.max()
    attention = np.zeros(len(features), dtype=np.float64)
    attention[mask] = np.exp(logits) / np.exp(logits).sum()
    weights = attention * np.maximum(positive - eta_positive, 0.0)
    weights *= np.maximum(differential - eta_differential, 0.0)
    weights[~mask] = 0.0
    if float(weights.sum()) <= mass_epsilon:
        return DifferentialWeights(None, text_distance, positive, differential, "no_positive_mass")
    return DifferentialWeights(weights, text_distance, positive, differential, "accepted")


def _timeline_bins(centers: np.ndarray, duration: float) -> np.ndarray:
    if duration <= 0 or centers.ndim != 1 or not len(centers):
        raise SupportError("canonical centers require a positive duration")
    if not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        raise SupportError("canonical centers must be finite and strictly increasing")
    if centers[0] < 0 or centers[-1] > duration:
        raise SupportError("canonical centers must lie inside the video")
    boundaries = np.empty(len(centers) + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[-1] = duration
    boundaries[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    return boundaries


def rebin_distribution(
    weights: np.ndarray,
    view_rf_intervals: np.ndarray,
    canonical_centers: np.ndarray,
    duration: float,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    raw = np.asarray(weights, dtype=np.float64)
    intervals = np.asarray(view_rf_intervals, dtype=np.float64)
    centers = np.asarray(canonical_centers, dtype=np.float64)
    if raw.ndim != 1 or intervals.shape != (len(raw), 2):
        raise SupportError("weights and view RF intervals have incompatible shapes")
    if not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0:
        raise SupportError("view weights must be finite, nonnegative and nonempty")
    if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]):
        raise SupportError("view RF intervals must have positive length")
    if np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] > duration):
        raise SupportError("view RF intervals must lie within the raw video")
    distribution = raw / raw.sum()
    bins = _timeline_bins(centers, duration)
    result = np.zeros(len(centers), dtype=np.float64)
    for mass, (left, right) in zip(distribution, intervals, strict=True):
        length = right - left
        overlap = np.maximum(0.0, np.minimum(bins[1:], right) - np.maximum(bins[:-1], left))
        result += mass * overlap / length
    if not np.isclose(result.sum(), 1.0, atol=tolerance, rtol=0):
        raise SupportError(f"rebin mass conservation failed: {result.sum():.12f}")
    return result


def jsd_stability(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise SupportError("JSD distributions must share one-dimensional shape")
    if np.any(left < 0) or np.any(right < 0):
        raise SupportError("JSD distributions must be nonnegative")
    if not np.isclose(left.sum(), 1.0) or not np.isclose(right.sum(), 1.0):
        raise SupportError("JSD distributions must sum to one")
    mean = 0.5 * (left + right)

    def kl(values: np.ndarray) -> float:
        keep = values > 0
        return float(np.sum(values[keep] * np.log(values[keep] / mean[keep])))

    jsd = 0.5 * (kl(left) + kl(right))
    stability = 1.0 - jsd / np.log(2.0)
    if stability < -1e-10 or stability > 1 + 1e-10:
        raise SupportError(f"JSD stability outside [0,1]: {stability}")
    return float(np.clip(stability, 0.0, 1.0))


def interval_union_length(intervals: np.ndarray) -> float:
    values = sorted((float(left), float(right)) for left, right in np.asarray(intervals))
    if not values:
        return 0.0
    left, right = values[0]
    if right <= left:
        raise SupportError("RF intervals must have positive length")
    total = 0.0
    for next_left, next_right in values[1:]:
        if next_right <= next_left:
            raise SupportError("RF intervals must have positive length")
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def select_support(
    distribution: np.ndarray,
    canonical_rf: np.ndarray,
    duration: float,
    *,
    target_mass: float = 0.60,
    max_clip_fraction: float = 0.40,
    min_retained_mass: float = 0.50,
    max_raw_rf_fraction: float = 0.50,
) -> SupportSelection:
    values = np.asarray(distribution, dtype=np.float64)
    intervals = np.asarray(canonical_rf, dtype=np.float64)
    if values.ndim != 1 or intervals.shape != (len(values), 2):
        raise SupportError("distribution/canonical RF shapes are incompatible")
    if duration <= 0 or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise SupportError("support distribution must sum to one over a positive duration")
    cap = int(np.floor(max_clip_fraction * len(values)))
    if cap < 1:
        return SupportSelection(None, 0.0, None, "too_few_clips")
    order = np.lexsort((np.arange(len(values)), -values))
    cumulative = np.cumsum(values[order])
    wanted = int(np.searchsorted(cumulative, target_mass, side="left")) + 1
    chosen = order[: min(wanted, cap)]
    retained = float(values[chosen].sum())
    if retained < min_retained_mass:
        return SupportSelection(None, retained, None, "diffuse_support")
    ratio = interval_union_length(intervals[chosen]) / duration
    if ratio > max_raw_rf_fraction:
        return SupportSelection(None, retained, ratio, "wide_rf")
    q = np.zeros_like(values)
    q[chosen] = values[chosen] / retained
    return SupportSelection(q, retained, ratio, "accepted")


def build_endpoint_support(
    view_features: tuple[np.ndarray, np.ndarray],
    view_validity: tuple[np.ndarray, np.ndarray],
    view_rf_intervals: tuple[np.ndarray, np.ndarray],
    view_ids: tuple[str, str],
    own_unit: np.ndarray,
    rival_unit: np.ndarray,
    canonical_centers: np.ndarray,
    canonical_rf: np.ndarray,
    duration: float,
    *,
    min_stability: float = 0.7,
    **thresholds: float,
) -> EndpointSupport:
    if view_ids[0] == view_ids[1]:
        return EndpointSupport(None, 0.0, None, 0.0, None, "no_distinct_views")
    weight_keys = {
        "h_min",
        "mass_epsilon",
        "tau_localization",
        "eta_positive",
        "eta_differential",
    }
    weight_options = {key: value for key, value in thresholds.items() if key in weight_keys}
    results = tuple(
        differential_support_weights(features, own_unit, rival_unit, valid, **weight_options)
        for features, valid in zip(view_features, view_validity, strict=True)
    )
    failed = next((item for item in results if item.reason != "accepted"), None)
    if failed is not None:
        return EndpointSupport(None, failed.text_distance, None, 0.0, None, failed.reason)
    rebinned = tuple(
        rebin_distribution(result.weights, intervals, canonical_centers, duration)
        for result, intervals in zip(results, view_rf_intervals, strict=True)
        if result.weights is not None
    )
    if len(rebinned) != 2:
        raise SupportError("accepted differential weights unexpectedly missing")
    stability = jsd_stability(rebinned[0], rebinned[1])
    if stability < min_stability:
        return EndpointSupport(None, results[0].text_distance, stability, 0.0, None, "low_stability")
    selection_keys = {
        "target_mass",
        "max_clip_fraction",
        "min_retained_mass",
        "max_raw_rf_fraction",
    }
    selection_options = {key: value for key, value in thresholds.items() if key in selection_keys}
    selection = select_support(
        0.5 * (rebinned[0] + rebinned[1]),
        canonical_rf,
        duration,
        **selection_options,
    )
    return EndpointSupport(
        selection.q,
        results[0].text_distance,
        stability,
        selection.retained_mass,
        selection.rf_union_ratio,
        selection.reason,
    )


def build_pair_support(
    endpoint_i_inputs: dict[str, object],
    endpoint_j_inputs: dict[str, object],
    unit_i: np.ndarray,
    unit_j: np.ndarray,
    semantic_reliability: float,
    *,
    min_stability: float = 0.7,
    **thresholds: float,
) -> PairSupport:
    if not 0 <= semantic_reliability <= 1:
        raise SupportError("semantic reliability must be in [0,1]")
    endpoint_i = build_endpoint_support(
        **endpoint_i_inputs,
        own_unit=unit_i,
        rival_unit=unit_j,
        min_stability=min_stability,
        **thresholds,
    )
    endpoint_j = build_endpoint_support(
        **endpoint_j_inputs,
        own_unit=unit_j,
        rival_unit=unit_i,
        min_stability=min_stability,
        **thresholds,
    )
    if endpoint_i.reason != "accepted":
        return PairSupport(endpoint_i, endpoint_j, 0.0, f"endpoint_i:{endpoint_i.reason}")
    if endpoint_j.reason != "accepted":
        return PairSupport(endpoint_i, endpoint_j, 0.0, f"endpoint_j:{endpoint_j.reason}")
    assert endpoint_i.stability is not None and endpoint_j.stability is not None
    reliability = semantic_reliability * min(endpoint_i.stability, endpoint_j.stability)
    return PairSupport(endpoint_i, endpoint_j, reliability, "accepted")
