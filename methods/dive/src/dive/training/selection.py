from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DevCandidate:
    epoch: int
    optimizer_step: int
    t2v_r1: float
    v2t_r1: float
    checkpoint_path: str

    @property
    def endpoint(self) -> float:
        return 0.5 * (self.t2v_r1 + self.v2t_r1)


def select_earliest_best(candidates: Sequence[DevCandidate], *, split: str) -> DevCandidate:
    if split != "dev":
        raise ValueError("checkpoint selection is dev-only")
    if not candidates:
        raise ValueError("checkpoint selection requires at least one candidate")
    for item in candidates:
        if item.epoch < 0 or item.optimizer_step < 0:
            raise ValueError("checkpoint epoch/step must be nonnegative")
        if not 0 <= item.t2v_r1 <= 1 or not 0 <= item.v2t_r1 <= 1:
            raise ValueError("recall metrics must be fractions in [0,1]")
    return min(candidates, key=lambda item: (-item.endpoint, item.optimizer_step, item.epoch))
