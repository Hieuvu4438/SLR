from __future__ import annotations

import math

import torch

from dive.losses import (
    directional_retrieval,
    local_loss_active,
    margins4,
    pair_loss_active,
    retrieval_loss,
)


def test_four_margin_order_is_two_rows_then_two_columns():
    quartet = torch.tensor([[5.0, 2.0], [1.0, 4.0]])
    torch.testing.assert_close(margins4(quartet), torch.tensor([3.0, 3.0, 4.0, 2.0]))


def test_multi_positive_retrieval_matches_logsumexp_oracle():
    scores = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float64)
    positives = torch.tensor([[True, True, False]])
    candidates = torch.ones_like(positives)
    actual = directional_retrieval(scores, positives, candidates, tau=1.0)
    expected = torch.logsumexp(scores[0], dim=0) - torch.logsumexp(scores[0, :2], dim=0)
    torch.testing.assert_close(actual, expected)


def test_reliability_uses_sampled_h_denominator_and_no_pooled_normalization():
    # Weighted mean for endpoint 0 cancels exactly; it must remain zero rather than normalize to 1.
    u = torch.tensor(
        [[[[1.0, 0.0], [-1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]],
        requires_grad=True,
    )
    q = torch.full((1, 2, 2), 0.5)
    d = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    h = torch.tensor([math.sqrt(2.0)])
    loss, quartet, _ = local_loss_active(u, q, d, torch.tensor([1.0]), h, 4)
    scaled, _, _ = local_loss_active(u, q, d, torch.tensor([0.25]), h, 4)
    assert quartet[0, 0, 0] == 0
    torch.testing.assert_close(scaled, loss * 0.25)
    loss.backward()
    assert u.grad is not None and bool((u.grad != 0).any())


def test_bridge_feasibility_excludes_only_margin_below_negative_two_gamma():
    baseline = torch.tensor([[[0.0, 0.21], [0.0, 0.01]]])
    current = baseline.clone().requires_grad_()
    loss, feasible = pair_loss_active(current, baseline, torch.ones(1), 1, gamma_train=0.1)
    assert feasible.tolist() == [[False, True, True, True]]
    assert loss.isfinite()
    loss.backward()
    assert current.grad is not None


def test_empty_active_auxiliary_losses_preserve_global_backward():
    scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    pc = torch.eye(2, dtype=torch.bool)
    candidates = torch.ones_like(pc)
    global_loss = retrieval_loss(scores, pc, candidates)
    empty_u = torch.empty((0, 2, 2, 3), requires_grad=True)
    local, _, _ = local_loss_active(
        empty_u,
        torch.empty((0, 2, 2)),
        torch.empty((0, 2, 3)),
        torch.empty(0),
        torch.empty(0),
        1,
    )
    empty_quartet = torch.empty((0, 2, 2), requires_grad=True)
    pair, feasible = pair_loss_active(empty_quartet, empty_quartet.detach(), torch.empty(0), 1)
    assert local.item() == pair.item() == 0 and feasible.shape == (0, 4)
    (global_loss + local + pair).backward()
    assert scores.grad is not None
