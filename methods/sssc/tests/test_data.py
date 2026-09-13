from __future__ import annotations

from method1.data import TrimmedDistributedGroupSampler, augment_caption


def test_stateless_augmentation_is_repeatable_and_epoch_scoped() -> None:
    text = "one two three four five six"
    first = augment_caption(text, seed=42, epoch=2, text_uid="t", probability=1.0)
    second = augment_caption(text, seed=42, epoch=2, text_uid="t", probability=1.0)
    assert first == second
    assert sorted(first.split()) == sorted(text.split())


def test_trimmed_sampler_has_no_cross_rank_duplicates() -> None:
    left = TrimmedDistributedGroupSampler(
        13, seed=42, global_batch_size=4, rank=0, world_size=2
    )
    right = TrimmedDistributedGroupSampler(
        13, seed=42, global_batch_size=4, rank=1, world_size=2
    )
    left.set_epoch(3)
    right.set_epoch(3)
    left_values = list(left)
    right_values = list(right)
    assert len(left_values) == len(right_values) == 6
    assert not set(left_values) & set(right_values)
    assert len(set(left_values + right_values)) == 12
    assert left.dropped_tail == 1
