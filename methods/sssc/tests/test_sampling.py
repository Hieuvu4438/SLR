from __future__ import annotations

import numpy as np
import torch

from method1.sampling import (
    choose_epoch_edits,
    cyclically_shift_support,
    legacy_feature_indices,
    mix_and_sample_features,
)


def test_feature_mix_weight_direction_and_padding() -> None:
    agnostic = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    aware = np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32)
    mixed, valid, indexes = mix_and_sample_features(
        agnostic, aware, agnostic_weight=0.9, feature_len=4
    )
    np.testing.assert_allclose(mixed[:2], [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(valid, [True, True, False, False])
    np.testing.assert_array_equal(indexes, [0, 1, -1, -1])
    np.testing.assert_array_equal(mixed[2:], 0.0)


def test_feature_linspace_is_exact() -> None:
    np.testing.assert_array_equal(legacy_feature_indices(5, 3), [0, 2, 4])


def test_epoch_edit_sampling_is_distinct_deterministic_and_padded() -> None:
    first, valid = choose_epoch_edits(["a", "b"], k=5, seed=42, epoch=3, text_uid="t")
    second, valid_again = choose_epoch_edits(
        ["a", "b"], k=5, seed=42, epoch=3, text_uid="t"
    )
    assert first == second
    assert len({item for item in first if item is not None}) == 2
    assert first[2:] == (None, None, None)
    np.testing.assert_array_equal(valid, [True, True, False, False, False])
    np.testing.assert_array_equal(valid, valid_again)


def test_random_support_shift_preserves_mass_and_avoids_identity() -> None:
    support = torch.tensor([[[[0.1, 0.2, 0.7, 0.0], [0.6, 0.3, 0.1, 0.0]]]])
    valid = torch.tensor([[True, True, True, False]])
    shifted = cyclically_shift_support(
        support,
        valid,
        seed=9,
        epoch=2,
        video_uids=["video:7"],
        edit_uids=[[["edit:a", "edit:b"]]],
    )
    torch.testing.assert_close(shifted.sum(-1), support.sum(-1))
    assert not torch.equal(shifted[..., :3], support[..., :3])
    assert not bool(shifted[..., 3].any())


def test_random_support_seed_uses_persistent_ids_and_one_clip_is_unchanged() -> None:
    support = torch.tensor(
        [
            [[ [0.1, 0.2, 0.3, 0.4] ]],
            [[ [1.0, 0.0, 0.0, 0.0] ]],
        ]
    )
    valid = torch.tensor([[True, True, True, True], [True, False, False, False]])
    kwargs = {
        "seed": 42,
        "epoch": 3,
        "video_uids": ["v0", "v1"],
        "edit_uids": [[["e0"]], [["e1"]]],
    }
    first = cyclically_shift_support(support, valid, **kwargs)
    second = cyclically_shift_support(support, valid, **kwargs)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first[0], support[0])
    torch.testing.assert_close(first[1], support[1])
