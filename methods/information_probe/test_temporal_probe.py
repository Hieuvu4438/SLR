import torch

from .temporal_probe import perturb


def test_temporal_permutations_preserve_padding_and_content():
    h=torch.arange(24.).reshape(2,4,3)
    valid=torch.tensor([[True,True,False,False],[True,True,True,True]])
    for mode in ('identity','reverse_input','shuffle_input'):
        changed,orders=perturb(h,valid,['a','b'],mode)
        torch.testing.assert_close(changed[~valid],h[~valid])
        for i,order in enumerate(orders):
            torch.testing.assert_close(changed[i,:len(order)][order.argsort()],h[i,:len(order)])
