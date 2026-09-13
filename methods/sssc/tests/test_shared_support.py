from __future__ import annotations

import pytest
import torch

from method1.losses.shared_support import reference_support, span_contrast_terms
from method1.schemas import SchemaError


def _fixture(dtype: torch.dtype = torch.float32):
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone()
    q_pos = torch.tensor([[[[1.0, 0.0]]]], dtype=dtype)
    q_neg = torch.tensor([[[[0.0, 1.0]]]], dtype=dtype)
    video_valid = torch.tensor([[True, True]])
    edit_valid = torch.tensor([[[True]]])
    confidence = torch.ones(1, 1, 1, dtype=dtype)
    return x, x_ref, q_pos, q_neg, video_valid, edit_valid, confidence


def test_support_mass_and_masking() -> None:
    x_ref = torch.tensor([[[1.0, 0.0], [123.0, -90.0], [0.0, 1.0]]])
    q = torch.tensor([[[[1.0, 0.0]]]])
    valid = torch.tensor([[True, False, True]])
    support = reference_support(x_ref, q, valid, tau=0.07)
    torch.testing.assert_close(support.sum(-1), torch.ones(1, 1, 1))
    assert support[..., 1].item() == 0.0


def test_shared_independent_algebraic_fixture() -> None:
    values = _fixture()
    shared = span_contrast_terms(*values, mode="shared", tau=0.07, margin=1.1)
    independent = span_contrast_terms(*values, mode="independent", tau=0.07, margin=1.1)
    assert shared.diagnostics["delta"].item() == pytest.approx(1.0, abs=2e-6)
    assert independent.diagnostics["delta"].item() == pytest.approx(0.0, abs=2e-6)


def test_active_and_inactive_hinge_have_expected_gradient() -> None:
    active_values = _fixture()
    active = span_contrast_terms(
        *active_values, mode="shared", tau=0.07, margin=1.1
    )
    active.numerator.backward()
    assert active.numerator.item() == pytest.approx(0.1, abs=2e-6)
    assert bool(active.diagnostics["active"].item()) is True
    assert active_values[0].grad is not None
    assert float(active_values[0].grad.abs().sum()) > 0

    inactive_values = _fixture()
    inactive = span_contrast_terms(
        *inactive_values, mode="shared", tau=0.07, margin=0.5
    )
    inactive.numerator.backward()
    assert inactive.numerator.item() == 0.0
    assert bool(inactive.diagnostics["active"].item()) is False
    torch.testing.assert_close(
        inactive_values[0].grad, torch.zeros_like(inactive_values[0]), atol=0, rtol=0
    )


def test_confidence_reduction_uses_one_weighted_numerator_and_denominator() -> None:
    values = list(_fixture())
    values[2] = values[2].repeat(1, 2, 1, 1)
    values[3] = values[3].repeat(1, 2, 1, 1)
    values[5] = values[5].repeat(1, 2, 1)
    values[6] = torch.tensor([[[0.25], [0.75]]])
    terms = span_contrast_terms(*values, mode="shared", tau=0.07, margin=1.1)
    assert terms.denominator.item() == 1.0
    assert terms.numerator.item() == pytest.approx(0.1, abs=2e-6)


def test_gradient_reaches_only_student_video() -> None:
    values = list(_fixture())
    values[2].requires_grad_(True)
    values[3].requires_grad_(True)
    terms = span_contrast_terms(*values, mode="shared", tau=0.07, margin=1.1)
    terms.numerator.backward()
    assert values[0].grad is not None and values[0].grad.abs().sum() > 0
    assert values[2].grad is None
    assert values[3].grad is None
    assert not terms.diagnostics["support"].requires_grad


def test_empty_edits_return_graph_connected_zero() -> None:
    values = list(_fixture())
    values[5] = torch.zeros_like(values[5])
    terms = span_contrast_terms(*values, mode="shared", tau=0.07, margin=0.05)
    assert terms.numerator.item() == 0.0
    assert terms.denominator.item() == 0.0
    terms.numerator.backward()
    assert values[0].grad is not None
    torch.testing.assert_close(values[0].grad, torch.zeros_like(values[0]))


def test_valid_zero_span_is_rejected() -> None:
    values = list(_fixture())
    values[2] = torch.zeros_like(values[2])
    with pytest.raises(SchemaError, match="nonzero"):
        span_contrast_terms(*values, mode="shared", tau=0.07, margin=0.05)


def test_raw_normalization_autograd_matches_finite_difference() -> None:
    values = list(_fixture(torch.float64))
    values[0] = torch.tensor(
        [[[0.8, 0.2], [0.1, 0.9]]], dtype=torch.float64, requires_grad=True
    )
    terms = span_contrast_terms(*values, mode="shared", tau=0.2, margin=2.0)
    gradient = torch.autograd.grad(terms.numerator, values[0])[0][0, 0, 1].item()
    epsilon = 1e-6
    observed = []
    for direction in (-1.0, 1.0):
        perturbed = values[0].detach().clone()
        perturbed[0, 0, 1] += direction * epsilon
        current = list(values)
        current[0] = perturbed
        observed.append(
            span_contrast_terms(*current, mode="shared", tau=0.2, margin=2.0)
            .numerator.item()
        )
    finite_difference = (observed[1] - observed[0]) / (2 * epsilon)
    assert gradient == pytest.approx(finite_difference, rel=1e-5, abs=1e-6)
