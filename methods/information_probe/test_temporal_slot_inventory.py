import pytest
import torch
from .temporal_slot_inventory import describe, slot_groups


def test_padding_and_signed_zero_adjacent_nonadjacent():
    a = torch.tensor([[0., 0.], [-0., 0.], [1., 2.], [0., 0.], [0., 0.]])
    valid = torch.tensor([True, True, True, True, False])
    assert slot_groups(a, valid) == [[0, 1, 3]]
    r = describe(a, valid, torch.arange(5), a, a)
    assert r['unique_n'] == 2 and r['excess_slot_n'] == 2
    assert r['repeated_pair_n'] == 3 and r['adjacent_repeated_pair_n'] == 1
    assert r['pair_causes'] == {'both_streams_equal': 3}
    a[0, 0] = float('nan')
    with pytest.raises(AssertionError):
        slot_groups(a, valid)


def test_fp16_only_merger_and_duplicate_index():
    a = torch.tensor([[1., 1.], [1.0001, 1.]])
    h = (1-.9)*a+.9*a
    valid = torch.ones(2, dtype=torch.bool)
    assert slot_groups(h, valid) == []
    assert describe(h.half(), valid, torch.arange(2), a, a)['pair_causes'] == {'fp16_only_merger': 1}
    h[:] = h[0].clone()
    assert describe(h, valid, torch.zeros(2, dtype=torch.long), a, a)['pair_causes'] == {'same_dense_index': 1}


def test_fusion_cancellation_and_single_slot():
    a = torch.tensor([[.9, .9], [0., 0.]])
    g = torch.tensor([[-.1, -.1], [0., 0.]])
    h = (1-.9)*a+.9*g
    valid = torch.ones(2, dtype=torch.bool)
    assert torch.equal(h[0], h[1])
    assert describe(h, valid, torch.arange(2), a, g)['pair_causes'] == {'fused_fp32_equal_not_both_streams': 1}
    r = describe(h, torch.tensor([True, False]), torch.tensor([0, -1]))
    assert r['eligible_pair_n'] == 0 and r['unique_n'] == 1
