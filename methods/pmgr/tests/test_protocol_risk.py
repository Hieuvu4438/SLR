from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pmgr.losses import objective_loss, pmgr_loss, whole_group_max


def _numeric_fixture():
    q = torch.tensor(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.7, 0.3],
            [0.4, 0.9, 0.2],
            [0.2, 0.0, 0.95],
            [0.1, 0.3, 0.85],
            [0.3, 0.2, 0.7],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    owner = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    ell = torch.tensor(0.6931471805599453, dtype=torch.float64, requires_grad=True)
    return q, owner, ell


def test_numeric_oracle_and_gradients():
    q, owner, ell = _numeric_fixture()
    output = pmgr_loss(q, owner, ell, 3, 6, rank_mix=0.25, rank_eta=0.1)
    expected_group = torch.tensor(
        [[0.8, 0.4, 0.3], [0.7, 0.9, 0.3], [0.3, 0.2, 0.95]], dtype=torch.float64
    )
    torch.testing.assert_close(output["group_scores"], expected_group, atol=0, rtol=0)
    expected = {
        "ce_t": 0.559557424510826,
        "ce_v": 0.667108048676086,
        "rank_t": 0.047084965497191,
        "rank_v": 0.182867776418063,
        "loss": 0.488743645184499,
    }
    for name, value in expected.items():
        assert float(output[name].detach()) == pytest.approx(value, abs=1e-8)
    dq, dell = torch.autograd.grad(output["loss"], (q, ell))
    assert float(dell.detach()) == pytest.approx(-0.235680363391833, abs=2e-6)
    expected_rows = torch.tensor(
        [[-0.1647319492, 0.0255603241, 0.0168597596],
         [-0.1112705270, 0.1955881752, 0.0858617589]], dtype=torch.float64
    )
    torch.testing.assert_close(dq[:2], expected_rows, atol=1e-9, rtol=1e-9)
    assert dq[1, 0] < 0  # weak non-winning positive still receives V2T supervision


def test_singleton_reduction_is_mixed_bidirectional_ce():
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    owner = torch.arange(4)
    ell = torch.tensor(0.4, dtype=torch.float64)
    actual = pmgr_loss(q, owner, ell, 4, 4, rank_mix=0.0)["loss"]
    target = torch.arange(4)
    scale = ell.exp()
    expected = 0.5 * (
        F.cross_entropy(scale * q, target) + F.cross_entropy(scale * q.T, target)
    )
    torch.testing.assert_close(actual, expected)


def test_population_weight_is_one_sixth_not_batch_video_mean():
    q = torch.zeros(5, 2, dtype=torch.float64)
    owner = torch.tensor([0, 0, 1, 1, 1])
    ell = torch.tensor(0.0, dtype=torch.float64)
    output = pmgr_loss(q, owner, ell, 10, 30)
    expected = (1 / 6) * F.cross_entropy(q, owner, reduction="sum")
    torch.testing.assert_close(output["ce_v"], expected)
    assert not torch.isclose(output["ce_v"], F.cross_entropy(q, owner))


def test_video_and_group_permutations_preserve_loss_with_remapping():
    q, owner, ell = _numeric_fixture()
    original = pmgr_loss(q, owner, ell, 3, 6, rank_mix=0.25, rank_eta=0.1)["loss"]
    row_order = torch.tensor([5, 2, 0, 4, 1, 3])
    reordered_rows = pmgr_loss(
        q[row_order], owner[row_order], ell, 3, 6, rank_mix=0.25, rank_eta=0.1
    )["loss"]
    torch.testing.assert_close(original, reordered_rows)
    column_order = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(column_order)
    inverse[column_order] = torch.arange(3)
    reordered_columns = pmgr_loss(
        q[:, column_order], inverse[owner], ell, 3, 6, rank_mix=0.25, rank_eta=0.1
    )["loss"]
    torch.testing.assert_close(original, reordered_columns)


def test_lambda_zero_has_no_hidden_rank_or_legacy_loss():
    q, owner, ell = _numeric_fixture()
    output = objective_loss("group_ce", q, owner, ell, 3, 6)
    direct = pmgr_loss(q, owner, ell, 3, 6, rank_mix=0.0)
    torch.testing.assert_close(output["loss"], direct["loss"])


def test_group_max_gradient_reaches_only_unique_winner():
    q, owner, _ = _numeric_fixture()
    reduced = whole_group_max(q, owner)
    derivative = torch.autograd.grad(reduced.sum(), q)[0]
    assert derivative[0, 0] == 1 and derivative[1, 0] == 0
    assert derivative[2, 1] == 1


def test_nonwinning_positive_still_receives_video_to_text_supervision():
    q, owner, ell = _numeric_fixture()
    output = pmgr_loss(q, owner, ell, 3, 6, rank_mix=0.0)
    total = torch.autograd.grad(output["loss"], q, retain_graph=True)[0]
    t2v_only = torch.autograd.grad(output["ce_t"], q)[0]
    assert t2v_only[1, 0] == 0
    assert total[1, 0] < 0


def test_gradcheck_away_from_ties():
    q, owner, ell = _numeric_fixture()
    assert torch.autograd.gradcheck(
        lambda scores, scale: pmgr_loss(scores, owner, scale, 3, 6, rank_mix=0.2)["loss"],
        (q, ell),
    )


def test_positive_set_controls_and_population_weighted_variant_are_distinct():
    q, owner, ell = _numeric_fixture()
    # Use population counts whose G/N ratio differs from this minibatch's B/Bv
    # ratio; otherwise the two V2T reductions legitimately have equal weights.
    dataset_groups, dataset_videos = 10, 30
    uniform = objective_loss("all_uniform_ce", q, owner, ell, dataset_groups, dataset_videos)
    set_positive = objective_loss("all_set_ce", q, owner, ell, dataset_groups, dataset_videos)
    population = objective_loss(
        "all_set_ce_population_weighted",
        q,
        owner,
        ell,
        dataset_groups,
        dataset_videos,
    )
    assert not torch.isclose(uniform["ce_t"], set_positive["ce_t"])
    torch.testing.assert_close(set_positive["ce_t"], population["ce_t"])
    expected_v = (dataset_groups / (dataset_videos * 3)) * F.cross_entropy(
        ell.exp() * q, owner, reduction="sum"
    )
    torch.testing.assert_close(population["ce_v"], expected_v)
    assert not torch.isclose(set_positive["ce_v"], population["ce_v"])


def test_c0_branch_loss_is_not_c1_mixed_score_loss():
    a = torch.tensor([[0.8, 0.1], [0.3, 0.6]], dtype=torch.float64)
    b = torch.tensor([[0.2, 0.7], [0.4, 0.9]], dtype=torch.float64)
    q = 0.5 * (a + b)
    owner = torch.arange(2)
    ell = torch.tensor(0.5, dtype=torch.float64)
    c0 = objective_loss("legacy_cico", q, owner, ell, 2, 2, a=a, b=b)["loss"]
    c1 = objective_loss("single_mixed_ce", q, owner, ell, 2, 2)["loss"]
    assert not torch.isclose(c0, c1)
