from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence


class SamplerError(ValueError):
    """A contrast schedule cannot satisfy the fixed effective-batch contract."""


class PairLike(Protocol):
    pair_id: str
    sample_i: str
    sample_j: str
    g: float


@dataclass(frozen=True)
class PlannedPair:
    pair_id: str
    sample_i: str
    sample_j: str
    active: bool


@dataclass(frozen=True)
class BatchPlanStep:
    step: int
    sample_ids: tuple[str, ...]
    pairs: tuple[PlannedPair, ...]

    @property
    def num_sampled(self) -> int:
        return len(self.pairs)

    @property
    def num_active(self) -> int:
        return sum(pair.active for pair in self.pairs)


@dataclass(frozen=True)
class BatchPlan:
    schema_version: str
    seed: int
    effective_batch_size: int
    contrasts_per_step: int
    endpoint_quota_per_epoch: int
    steps: tuple[BatchPlanStep, ...]
    fingerprint: str
    unused_pair_ids: tuple[str, ...]
    quota_rejected_pair_ids: tuple[str, ...]
    endpoint_exposure: dict[str, int]
    ordinary_exposure: dict[str, int]


def _plan_payload(
    sample_ids: Sequence[str],
    pairs: Sequence[PairLike],
    seed: int,
    batch_size: int,
    contrasts_per_step: int,
    endpoint_quota: int,
) -> dict[str, object]:
    return {
        "schema_version": "batch_plan_inputs.v1",
        "sample_ids": list(sample_ids),
        "pairs": [
            [pair.pair_id, pair.sample_i, pair.sample_j, float(pair.g)]
            for pair in sorted(pairs, key=lambda item: item.pair_id)
        ],
        "seed": seed,
        "batch_size": batch_size,
        "contrasts_per_step": contrasts_per_step,
        "endpoint_quota": endpoint_quota,
    }


def build_batch_plan(
    sample_ids: Sequence[str],
    pairs: Sequence[PairLike],
    *,
    seed: int,
    effective_batch_size: int = 128,
    contrasts_per_step: int = 16,
    endpoint_quota_per_epoch: int = 4,
) -> BatchPlan:
    samples = tuple(map(str, sample_ids))
    if len(samples) != len(set(samples)):
        raise SamplerError("train sample IDs must be unique")
    if effective_batch_size <= 0 or len(samples) < effective_batch_size:
        raise SamplerError("train set must contain at least one unique effective batch")
    if contrasts_per_step < 0 or 2 * contrasts_per_step > effective_batch_size:
        raise SamplerError("2*contrasts_per_step cannot exceed effective batch size")
    if endpoint_quota_per_epoch <= 0:
        raise SamplerError("endpoint quota must be positive")
    sample_set = set(samples)
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise SamplerError("pair IDs must be unique")
    unordered: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair.sample_i == pair.sample_j or {pair.sample_i, pair.sample_j} - sample_set:
            raise SamplerError(f"invalid endpoints for pair {pair.pair_id}")
        key = tuple(sorted((pair.sample_i, pair.sample_j)))
        if key in unordered:
            raise SamplerError(f"duplicate unordered pair: {key}")
        unordered.add(key)
    payload = _plan_payload(
        samples,
        pairs,
        seed,
        effective_batch_size,
        contrasts_per_step,
        endpoint_quota_per_epoch,
    )
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rng = random.Random(seed)
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)
    ordinary_order = list(samples)
    rng.shuffle(ordinary_order)
    endpoint_exposure = {sample: 0 for sample in samples}
    ordinary_exposure = {sample: 0 for sample in samples}
    pair_cursor = 0
    ordinary_cursor = 0
    steps: list[BatchPlanStep] = []
    quota_rejected: list[str] = []
    for step_index in range(math.ceil(len(samples) / effective_batch_size)):
        selected_pairs: list[PairLike] = []
        while pair_cursor < len(shuffled_pairs) and len(selected_pairs) < contrasts_per_step:
            pair = shuffled_pairs[pair_cursor]
            pair_cursor += 1
            if (
                endpoint_exposure[pair.sample_i] >= endpoint_quota_per_epoch
                or endpoint_exposure[pair.sample_j] >= endpoint_quota_per_epoch
            ):
                quota_rejected.append(pair.pair_id)
                continue
            selected_pairs.append(pair)
            endpoint_exposure[pair.sample_i] += 1
            endpoint_exposure[pair.sample_j] += 1
        batch_ids: list[str] = []
        for pair in selected_pairs:
            if pair.sample_i not in batch_ids:
                batch_ids.append(pair.sample_i)
            if pair.sample_j not in batch_ids:
                batch_ids.append(pair.sample_j)
        attempts = 0
        while len(batch_ids) < effective_batch_size:
            sample = ordinary_order[ordinary_cursor % len(ordinary_order)]
            ordinary_cursor += 1
            attempts += 1
            if sample not in batch_ids:
                batch_ids.append(sample)
                ordinary_exposure[sample] += 1
            if attempts > len(samples) * 2:
                raise SamplerError("could not fill a unique effective batch")
        planned = tuple(
            PlannedPair(pair.pair_id, pair.sample_i, pair.sample_j, bool(pair.g > 0))
            for pair in selected_pairs
        )
        steps.append(BatchPlanStep(step_index, tuple(batch_ids), planned))
    unused = tuple(sorted(pair.pair_id for pair in shuffled_pairs[pair_cursor:]))
    return BatchPlan(
        schema_version="batch_plan.v1",
        seed=seed,
        effective_batch_size=effective_batch_size,
        contrasts_per_step=contrasts_per_step,
        endpoint_quota_per_epoch=endpoint_quota_per_epoch,
        steps=tuple(steps),
        fingerprint=fingerprint,
        unused_pair_ids=unused,
        quota_rejected_pair_ids=tuple(sorted(quota_rejected)),
        endpoint_exposure=endpoint_exposure,
        ordinary_exposure=ordinary_exposure,
    )


def write_batch_plan(plan: BatchPlan, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(plan), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def require_common_plan(first: BatchPlan, second: BatchPlan) -> None:
    if first.fingerprint != second.fingerprint or first.steps != second.steps:
        raise SamplerError("control variants do not share the same batch plan")
