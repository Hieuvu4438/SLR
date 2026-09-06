from __future__ import annotations

from elsc.data.tokenize import augment_caption, random_swap_words


class _FixedRng:
    def __init__(self, values: list[int]):
        self.values = iter(values)

    def randint(self, _left: int, _right: int) -> int:
        return next(self.values)


def test_random_swap_words_performs_one_swap() -> None:
    assert random_swap_words("one two three", _FixedRng([0, 2])) == "three two one"


def test_augmentation_is_stable_for_sample_and_epoch() -> None:
    first = augment_caption("one two three four", "pair-7", seed=42, epoch=3)
    second = augment_caption("one two three four", "pair-7", seed=42, epoch=3)
    assert first == second


def test_short_caption_is_unchanged() -> None:
    assert augment_caption("only", "pair-8", seed=42, epoch=0) == "only"
