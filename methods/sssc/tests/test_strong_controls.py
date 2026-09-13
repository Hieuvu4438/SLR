from __future__ import annotations

import pytest
import torch

from method1.losses.strong_controls import (
    caption_hard_negative_terms,
    fsc_local_hard_negative_terms,
)


def _fixture():
    video = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.8, 0.2], [0.1, 0.9]],
        ],
        requires_grad=True,
    )
    video_valid = torch.tensor([[True, True], [True, True]])
    captions = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.9, 0.1]],
                [[0.0, 1.0], [0.1, 0.9]],
                [[100.0, -100.0], [100.0, -100.0]],
            ],
            [
                [[0.0, 1.0], [0.1, 0.9]],
                [[1.0, 0.0], [0.9, 0.1]],
                [[-100.0, 100.0], [-100.0, 100.0]],
            ],
        ],
        requires_grad=True,
    )
    caption_valid = torch.ones(2, 3, 2, dtype=torch.bool)
    negative_valid = torch.tensor([[True, False], [False, False]])
    return video, video_valid, captions, caption_valid, negative_valid


def test_caption_hard_negative_masks_invalid_classes_and_updates_student_text() -> None:
    values = _fixture()
    terms = caption_hard_negative_terms(
        *values, logit_scale=torch.tensor(0.0), temperature=0.07
    )
    assert terms.denominator.item() == 1.0
    assert torch.isfinite(terms.numerator)
    assert torch.isneginf(terms.diagnostics["candidate_logits"][0, 2])
    terms.numerator.backward()
    assert values[0].grad is not None and values[0].grad.abs().sum() > 0
    assert values[2].grad is not None and values[2].grad.abs().sum() > 0
    assert values[2].grad[0, 2].abs().sum().item() == 0.0


def test_fsc_minmax_uniform_fallback_is_finite_and_mass_preserving() -> None:
    video = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [9.0, 9.0]]], requires_grad=True)
    captions = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.0, 1.0]]]],
        requires_grad=True,
    )
    terms = fsc_local_hard_negative_terms(
        video,
        torch.tensor([[True, True, False]]),
        captions,
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.tensor([[True]]),
        logit_scale=torch.tensor(0.0),
        loss_name="focal_loss",
        focal_gamma=1.0,
        label_smoothing=0.1,
    )
    support = terms.diagnostics["candidate_support"]
    torch.testing.assert_close(support.sum(-1), torch.ones_like(support.sum(-1)))
    torch.testing.assert_close(support[..., :2], torch.full_like(support[..., :2], 0.5))
    assert not bool(support[..., 2].any())
    terms.numerator.backward()
    assert torch.isfinite(video.grad).all()
    assert torch.isfinite(captions.grad).all()


def test_san_inspired_fixture_matches_token_softmax_then_clip_mean() -> None:
    video = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    captions = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]
        ]
    )
    tau = 0.07
    terms = caption_hard_negative_terms(
        video,
        torch.ones(1, 2, dtype=torch.bool),
        captions,
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        logit_scale=torch.tensor(0.0),
        temperature=tau,
    )
    original_score = torch.sigmoid(torch.tensor(1.0 / tau))
    negative_score = torch.tensor(0.5)
    expected = torch.nn.functional.softplus(negative_score - original_score)
    torch.testing.assert_close(terms.numerator, expected)
    assert terms.denominator.item() == 1.0


def test_fsc_source_style_fixture_matches_detached_minmax_focal_value() -> None:
    video = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    captions = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    terms = fsc_local_hard_negative_terms(
        video,
        torch.ones(1, 2, dtype=torch.bool),
        captions,
        torch.ones(1, 2, 1, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        logit_scale=torch.tensor(0.0),
        loss_name="focal_loss",
        focal_gamma=1.0,
        label_smoothing=0.1,
    )
    # Both candidate-local matches are exactly one. Equal class probabilities therefore
    # yield focal factor .5; the smoothed target still has total mass one.
    expected = 0.5 * torch.log(torch.tensor(2.0))
    torch.testing.assert_close(terms.numerator, expected)
    torch.testing.assert_close(
        terms.diagnostics["candidate_support"],
        torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
    )


@pytest.mark.parametrize("control", [caption_hard_negative_terms, fsc_local_hard_negative_terms])
def test_strong_control_all_invalid_negatives_returns_graph_zero(control) -> None:
    values = list(_fixture())
    values[4] = torch.zeros_like(values[4])
    kwargs = {"logit_scale": torch.tensor(0.0)}
    if control is fsc_local_hard_negative_terms:
        kwargs.update(
            {"loss_name": "cross_entropy", "focal_gamma": 1.0, "label_smoothing": 0.1}
        )
    terms = control(*values, **kwargs)
    assert terms.denominator.item() == 0.0
    assert terms.numerator.item() == 0.0
    terms.numerator.backward()
    assert values[0].grad is not None
    assert values[2].grad is not None
    torch.testing.assert_close(values[0].grad, torch.zeros_like(values[0]))
