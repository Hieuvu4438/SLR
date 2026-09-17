import numpy as np
import torch

from .common import ranks
from .scoring import Readout, balanced_loss, channels


def inputs():
    torch.manual_seed(3)
    return (torch.randn(3, 5, 512), torch.randn(3, 4, 512),
            torch.tensor([[1, 0, 0, 0, 1]]*3), torch.tensor([[1, 1, 1, 0]]*3))


def test_permutation_invariance():
    v, t, vm, tm = inputs()
    for masked in (False, True):
        a = channels(v, t, vm, tm, 10., masked=masked)
        b = channels(v.flip(1), t.flip(1), vm.flip(1), tm.flip(1), 10., masked=masked)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)


def test_masked_padding_invariance():
    v, t, vm, tm = inputs()
    a = channels(v, t, vm, tm, 10., masked=True)
    v[vm != 0] = torch.randn_like(v[vm != 0])*10
    t[tm != 1] = torch.randn_like(t[tm != 1])*10
    for x, y in zip(a, channels(v, t, vm, tm, 10., masked=True)):
        torch.testing.assert_close(x, y)


def test_readout_zero_init_and_capacity_control():
    counts = []
    for mode in ('interaction', 'pooled', 'zero'):
        m = Readout(mode)
        counts.append(sum(p.numel() for p in m.parameters()))
        a, b = m(*inputs())
        assert torch.count_nonzero(a) == torch.count_nonzero(b) == 0
        balanced_loss(a, b).backward()
        assert all(p.grad is not None for p in m.parameters())
    assert len(set(counts)) == 1


def test_orientation_and_ties():
    s = np.array([[2., 1., 3.], [1., 2., 1.], [0., 0., 2.]], dtype=np.float32)
    r = ranks(s)
    assert r['T2V'].tolist() == [0, 0, 1]
    assert r['V2T'].tolist() == [1, 0, 0]
