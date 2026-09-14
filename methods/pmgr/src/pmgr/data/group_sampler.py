from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from slr_common.utils import stable_seed


class GroupBatchSampler(Sampler[list[int]]):
    """Uniform group permutation with a logged, unpadded dropped tail."""

    def __init__(self, group_count: int, batch_size: int, *, seed: int, drop_last: bool = True):
        if group_count < 2 or batch_size < 2 or batch_size > group_count:
            raise ValueError("group sampler requires 2 <= batch_size <= group_count")
        self.group_count = int(group_count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.cursor = 0
        self._last_permutation: list[int] | None = None

    def set_epoch(self, epoch: int, *, cursor: int = 0) -> None:
        if epoch < 0 or cursor < 0 or cursor > len(self):
            raise ValueError("invalid sampler epoch or cursor")
        self.epoch = int(epoch)
        self.cursor = int(cursor)

    def permutation(self, *, epoch: int | None = None) -> list[int]:
        selected_epoch = self.epoch if epoch is None else int(epoch)
        if selected_epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        generator = np.random.default_rng(
            stable_seed(self.seed, selected_epoch, "pmgr_group_order")
        )
        return [int(value) for value in generator.permutation(self.group_count)]

    @property
    def dropped_indices(self) -> list[int]:
        order = self._last_permutation or self.permutation()
        remainder = self.group_count % self.batch_size
        return order[-remainder:] if self.drop_last and remainder else []

    def __len__(self) -> int:
        quotient, remainder = divmod(self.group_count, self.batch_size)
        return quotient if self.drop_last or remainder == 0 else quotient + 1

    def __iter__(self) -> Iterator[list[int]]:
        order = self.permutation()
        self._last_permutation = order
        usable = self.group_count
        if self.drop_last:
            usable -= self.group_count % self.batch_size
        batches = [order[start : min(start + self.batch_size, usable)] for start in range(0, usable, self.batch_size)]
        for batch in batches[self.cursor :]:
            if len(batch) < 2:
                raise ValueError("production group batch needs at least two groups")
            yield batch
