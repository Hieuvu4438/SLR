import torch

from methods.information_probe.mask_factor_probe import factor_channels
from methods.information_probe.scoring import channels


def test_factor_masks_channel_isolation_and_joint_parity():
    gen = torch.Generator().manual_seed(42)
    v, t = torch.randn(2, 4, 8, generator=gen), torch.randn(3, 5, 8, generator=gen)
    vm = torch.tensor([[1, 0, 0, 1], [1, 0, 0, 0]])
    tm = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    args = v, t, vm, tm, 3.
    base = factor_channels(*args)
    for a, b in zip(base, channels(*args)):
        torch.testing.assert_close(a, b)
    for flags in ({'exclude_cls': True}, {'exclude_vpad': True}, {'exclude_cls': True, 'exclude_vpad': True}):
        torch.testing.assert_close(factor_channels(*args, **flags)[0], base[0])
    torch.testing.assert_close(factor_channels(*args, exclude_tpad=True)[1], base[1])
    joint = factor_channels(*args, exclude_cls=True, exclude_vpad=True, exclude_tpad=True)
    for a, b in zip(joint, channels(*args, masked=True)):
        torch.testing.assert_close(a, b)
