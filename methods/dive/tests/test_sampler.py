from __future__ import annotations

from dataclasses import dataclass

import pytest

from dive.training.sampler import SamplerError, build_batch_plan, require_common_plan


@dataclass(frozen=True)
class Pair:
    pair_id: str
    sample_i: str
    sample_j: str
    g: float


def _samples() -> list[str]:
    return [f"s{index}" for index in range(10)]


def _pairs() -> list[Pair]:
    return [
        Pair("p0", "s0", "s1", 0.8),
        Pair("p1", "s2", "s3", 0.0),
        Pair("p2", "s4", "s5", 0.7),
    ]


def test_plan_has_unique_batches_and_h_counts_failed_support_before_rejection():
    plan = build_batch_plan(
        _samples(),
        _pairs(),
        seed=17,
        effective_batch_size=6,
        contrasts_per_step=3,
        endpoint_quota_per_epoch=4,
    )
    assert len(plan.steps) == 2
    assert all(len(step.sample_ids) == len(set(step.sample_ids)) == 6 for step in plan.steps)
    scheduled = [pair for step in plan.steps for pair in step.pairs]
    assert {pair.pair_id for pair in scheduled} == {"p0", "p1", "p2"}
    step_with_failed = next(step for step in plan.steps if any(pair.pair_id == "p1" for pair in step.pairs))
    assert step_with_failed.num_sampled == 3
    assert step_with_failed.num_active == 2
    assert {"s2", "s3"}.issubset(step_with_failed.sample_ids)


def test_endpoint_quota_is_enforced_and_reported():
    pairs = [Pair(f"p{index}", "s0", f"s{index + 1}", 1.0) for index in range(4)]
    plan = build_batch_plan(
        _samples(),
        pairs,
        seed=23,
        effective_batch_size=5,
        contrasts_per_step=2,
        endpoint_quota_per_epoch=1,
    )
    assert plan.endpoint_exposure["s0"] == 1
    assert len(plan.quota_rejected_pair_ids) == 3
    assert sum(step.num_sampled for step in plan.steps) == 1


def test_plan_is_reproducible_and_controls_require_exact_fingerprint():
    kwargs = dict(
        sample_ids=_samples(),
        pairs=_pairs(),
        effective_batch_size=6,
        contrasts_per_step=2,
        endpoint_quota_per_epoch=2,
    )
    first = build_batch_plan(seed=17, **kwargs)
    replay = build_batch_plan(seed=17, **kwargs)
    require_common_plan(first, replay)
    changed = build_batch_plan(seed=18, **kwargs)
    with pytest.raises(SamplerError, match="same batch plan"):
        require_common_plan(first, changed)


def test_invalid_duplicate_or_undersized_inputs_fail_closed():
    with pytest.raises(SamplerError, match="at least one unique"):
        build_batch_plan(
            ["s0", "s1"],
            [Pair("p", "s0", "s1", 1.0)],
            seed=1,
            effective_batch_size=3,
        )
    with pytest.raises(SamplerError, match="duplicate unordered"):
        build_batch_plan(
            _samples(),
            [Pair("p0", "s0", "s1", 1.0), Pair("p1", "s1", "s0", 1.0)],
            seed=1,
            effective_batch_size=5,
            contrasts_per_step=1,
        )
