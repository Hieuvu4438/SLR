from __future__ import annotations

import pytest
import torch

from pmgr.scoring import (
    ScoringContractError,
    mixed_pair_scores,
    mixed_pair_scores_blocked,
    video_valid_from_legacy,
)
from pmgr.losses import whole_group_max


def _explicit_scores(video, text, aug, video_valid, text_valid, aug_valid, omega, sigma):
    video = torch.nn.functional.normalize(video, dim=-1, eps=1e-6)
    text = torch.nn.functional.normalize(text, dim=-1, eps=1e-6)
    aug = torch.nn.functional.normalize(aug, dim=-1, eps=1e-6)
    a = torch.empty((len(video), len(text)), dtype=video.dtype)
    b = torch.empty_like(a)
    for row in range(len(video)):
        for column in range(len(text)):
            va = video[row, video_valid[row]]
            ta = text[column, text_valid[column]]
            ca = va @ ta.T
            a[row, column] = (torch.softmax(ca / sigma, dim=1) * ca).sum(1).mean()
            tb = aug[column, aug_valid[column]]
            cb = va @ tb.T
            b[row, column] = (torch.softmax(cb / sigma, dim=0) * cb).sum(0).mean()
    return omega * a + (1 - omega) * b, a, b


def _fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    video = torch.randn(5, 4, 3, generator=generator, dtype=dtype, requires_grad=True)
    text = torch.randn(3, 5, 3, generator=generator, dtype=dtype, requires_grad=True)
    aug = torch.randn(3, 6, 3, generator=generator, dtype=dtype, requires_grad=True)
    video_valid = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]],
        dtype=torch.bool,
    )
    text_valid = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=torch.bool
    )
    aug_valid = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 0]], dtype=torch.bool
    )
    return video, text, aug, video_valid, text_valid, aug_valid


def test_orientation_and_explicit_valid_position_loop():
    values = _fixture()
    actual = mixed_pair_scores(*values, omega=0.3, sigma=0.11)
    expected = _explicit_scores(*values, 0.3, 0.11)
    assert actual[0].shape == (5, 3)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_padding_values_and_gradients_are_invariant():
    values = _fixture()
    q1, _, _ = mixed_pair_scores(*values)
    gradients1 = torch.autograd.grad(q1.sum(), values[:3], retain_graph=False)
    changed = [value.detach().clone().requires_grad_(True) for value in values[:3]]
    masks = values[3:]
    for hidden, valid in zip(changed, masks, strict=True):
        hidden.data[~valid] = 1e6
    q2, _, _ = mixed_pair_scores(*changed, *masks)
    gradients2 = torch.autograd.grad(q2.sum(), changed)
    torch.testing.assert_close(q1, q2)
    for left, right, valid in zip(gradients1, gradients2, masks, strict=True):
        torch.testing.assert_close(left[valid], right[valid])
        assert torch.count_nonzero(right[~valid]) == 0


def test_chunk_invariance():
    values = _fixture(dtype=torch.float32)
    direct = mixed_pair_scores(*values)
    for video_block in (1, 2, 5):
        for text_block in (1, 2, 3):
            blocked = mixed_pair_scores_blocked(
                *values, video_block=video_block, text_block=text_block
            )
            for left, right in zip(direct, blocked, strict=True):
                torch.testing.assert_close(left, right)


def test_no_valid_position_fails():
    values = list(_fixture())
    values[3][0] = False
    with pytest.raises(ScoringContractError, match="valid positions"):
        mixed_pair_scores(*values)


def test_mask_conversion_excludes_cls_and_padding():
    mask = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]])
    expected = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
    assert torch.equal(video_valid_from_legacy(mask, hidden_length=4), expected)
    with pytest.raises(ScoringContractError):
        video_valid_from_legacy(1 - mask[:, :1])


def test_scale_is_not_part_of_unscaled_pair_score():
    values = _fixture()
    q, _, _ = mixed_pair_scores(*values)
    ell = torch.tensor(0.2, dtype=q.dtype, requires_grad=True)
    assert torch.autograd.grad(q.sum(), ell, allow_unused=True)[0] is None


def test_group_max_is_after_mixture_not_mixture_of_channel_maxima():
    a = torch.tensor([[0.9], [0.1]])
    b = torch.tensor([[0.1], [0.9]])
    owner = torch.tensor([0, 0])
    mixed_max = whole_group_max(0.5 * a + 0.5 * b, owner)
    wrong = 0.5 * whole_group_max(a, owner) + 0.5 * whole_group_max(b, owner)
    assert mixed_max.item() == pytest.approx(0.5)
    assert wrong.item() == pytest.approx(0.9)


def test_declared_legacy_mode_differs_only_by_inner_padding_policy():
    values = _fixture()
    corrected = mixed_pair_scores(*values, mask_policy="valid_tokens_only")[0]
    legacy = mixed_pair_scores(*values, mask_policy="legacy_unmasked")[0]
    assert not torch.allclose(corrected, legacy)
