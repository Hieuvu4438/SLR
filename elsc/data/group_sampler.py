from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from elsc.data.manifest import ManifestRecord
from elsc.utils import stable_seed


class CaptionGroupSampler(Sampler[int]):
    """Choose one video per caption group per epoch, then shuffle the groups."""

    def __init__(self, records: Sequence[ManifestRecord], *, seed: int):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[record.caption_id].append(index)
        if not groups:
            raise ValueError("caption-group sampler requires at least one record")
        self.groups = dict(groups)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[int]:
        selected = []
        for caption_id, indexes in self.groups.items():
            generator = np.random.default_rng(
                stable_seed(self.seed, self.epoch, caption_id, "caption_group_member")
            )
            selected.append(indexes[int(generator.integers(0, len(indexes)))])
        order = np.random.default_rng(
            stable_seed(self.seed, self.epoch, "caption_group_order")
        ).permutation(len(selected))
        return iter(selected[int(index)] for index in order)
