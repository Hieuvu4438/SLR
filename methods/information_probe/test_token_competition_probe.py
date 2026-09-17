import torch
from .token_competition_probe import reduce_channels


def test_endpoint_order_and_legacy_inner_padding():
    a = torch.tensor([[[[.1, .9], [.2, .8]]]])
    vm, tm = torch.tensor([[1, 0]]), torch.tensor([[1, 0]])
    hard = reduce_channels(a, vm, tm, 1., 'hard')
    soft = reduce_channels(a, vm, tm, 1., 'soft')
    mean = reduce_channels(a, vm, tm, 1., 'mean')
    # PAD text remains eligible in inner A even though excluded by outer B.
    assert torch.allclose(hard[0], torch.tensor([[.8]]))
    assert torch.allclose(hard[1], torch.tensor([[.2]]))
    for m, s, h in zip(mean, soft, hard):
        assert (m <= s).all() and (s <= h).all()


def test_constant_tensor_endpoints_equal():
    a = torch.full((2, 3, 4, 5), .3)
    vm, tm = torch.zeros(2, 4), torch.ones(3, 5)
    for mode in ('soft', 'hard', 'mean'):
        assert all(x.shape == (2, 3) and torch.allclose(x, torch.full((2, 3), .6))
                   for x in reduce_channels(a, vm, tm, 2., mode))


def test_soft_expectation_limits():
    a = torch.tensor([.1, .4, .8], dtype=torch.float64)
    assert torch.allclose((a*(a/1e-5).softmax(0)).sum(), a.max())
    assert torch.allclose((a*(a/1e8).softmax(0)).sum(), a.mean())
