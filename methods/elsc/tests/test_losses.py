from __future__ import annotations

import math

import torch

from elsc.losses.coarse import balanced_clcl_loss, diagonal_cross_entropy
from elsc.losses.distillation import bidirectional_kl
from elsc.losses.evidence import (
    evidence_losses,
    huber,
    receptive_field_closure,
    select_matched_control,
)
from elsc.losses.lexical import globally_normalized_auxiliary, lexical_loss
from elsc.train import _in_batch_word_negative_mask


def test_balanced_coarse_exact_formula():
    a = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    b = torch.tensor([[2.0, 0.0], [5.0, 1.0]])
    eta = 0.3
    expected = 0.5 * (
        eta * diagonal_cross_entropy(a)
        + (1 - eta) * diagonal_cross_entropy(a.T)
        + eta * diagonal_cross_entropy(b.T)
        + (1 - eta) * diagonal_cross_entropy(b)
    )
    assert torch.allclose(balanced_clcl_loss(a, b, dual_mix=eta), expected)


def test_lexical_equal_scores_and_masked_negative_math():
    z = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    weights = torch.tensor([[0.5, 0.5]])
    pos = torch.tensor([[1.0, 0.0]])
    neg = torch.tensor([[[1.0, 0.0], [100.0, 0.0]]])
    valid = torch.tensor([[True, False]])
    rho = torch.tensor([0.4])
    numerator, count = lexical_loss(z, weights, pos, neg, valid, rho, margin=0.1, temperature=0.2)
    expected = 0.4 * math.log(1 + math.exp(0.1 / 0.2))
    assert count.item() == 1
    assert math.isclose(numerator.item(), expected, rel_tol=1e-6)
    numerator.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_empty_lexical_is_differentiable():
    z = torch.empty(0, 4, 8, requires_grad=True)
    value, count = lexical_loss(
        z,
        torch.empty(0, 4),
        torch.empty(0, 8),
        torch.empty(0, 5, 8),
        torch.empty(0, 5, dtype=torch.bool),
        torch.empty(0),
    )
    value.backward()
    assert count.item() == 0 and z.grad is not None


def test_evidence_sign_and_huber_scale():
    dep, inv = evidence_losses(
        torch.tensor([0.10]),
        torch.tensor([0.07]),
        torch.tensor([0.10]),
        torch.tensor([1.0]),
        dependence_margin=0.02,
    )
    assert dep.item() == 0 and inv.item() == 0
    values = huber(torch.tensor([0.0, 0.025, 0.1]), 0.05)
    assert torch.allclose(values, torch.tensor([0.0, 0.5 * 0.025**2, 0.05 * (0.1 - 0.025)]))


def test_kl_direction_detaches_teacher_and_is_zero_at_identity():
    teacher = torch.randn(4, 4, requires_grad=True)
    student = teacher.detach().clone().requires_grad_(True)
    loss = bidirectional_kl(teacher, student)
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert teacher.grad is None and student.grad is not None


def test_rf_closure_and_matched_control_preserve_token_count():
    starts = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0])
    ends = torch.tensor([3.0, 7.0, 11.0, 15.0, 19.0])
    valid = torch.ones(5, dtype=torch.bool)
    evidence = receptive_field_closure(starts, ends, (0.5, 4.5), valid)
    # Token 1 is included even though its center is outside the interval.
    assert evidence.tolist() == [True, True, False, False, False]
    control = select_matched_control(
        starts,
        ends,
        valid,
        evidence,
        torch.zeros_like(valid),
        seed=42,
        pair_id="p",
        target_id=1,
        duration_tolerance=0.01,
    )
    assert control is not None
    assert int(control.sum()) == int(evidence.sum())
    assert not bool((control & evidence).any())


def test_ddp_local_sum_scaling_matches_global_mean_algebra():
    parameter = torch.tensor(2.0, requires_grad=True)
    # Rank sums are 2*p (two records) and 1*p (one record). DDP averages
    # gradients, so each rank scales its local sum by world_size/N_global.
    rank0 = globally_normalized_auxiliary(2 * parameter, torch.tensor(3), world_size=2)
    rank1 = globally_normalized_auxiliary(parameter, torch.tensor(3), world_size=2)
    averaged = 0.5 * (rank0 + rank1)
    averaged.backward()
    assert parameter.grad.item() == 1.0


def test_local_word_video_mask_excludes_self_and_same_word_occurrences():
    mask = _in_batch_word_negative_mask(torch.tensor([3, 7, 3, 9]))
    assert mask.tolist() == [
        [False, True, False, True],
        [True, False, True, True],
        [False, True, False, True],
        [True, True, True, False],
    ]
