import pytest
import torch

from methods.information_probe.retention_probe import paired_moments, ridge_map, predict, reconstruction, replace_valid


def test_weighted_affine_recovery_and_centered_moments():
    x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.double)
    y = x@torch.tensor([[2., -1.], [3., 4.]], dtype=torch.double)+7
    weight = torch.tensor([.1, .2, .3, .4], dtype=torch.double)
    mx, my, xx, yy, xy = paired_moments(x, y, weight, chunk=2)
    torch.testing.assert_close(mx, (x*weight[:, None]).sum(0))
    torch.testing.assert_close(xy, (x-mx).T@((y-my)*weight[:, None]))
    torch.testing.assert_close(predict(x, ridge_map(mx, my, xx, xy, penalty=0)), y)
    torch.testing.assert_close(predict(y, ridge_map(my, mx, yy, xy.T, penalty=0)), x)
    with pytest.raises(ValueError):
        paired_moments(x, y, weight*2)


def test_reconstruction_equal_sequence_weight_and_padding_preservation():
    x = torch.zeros(2, 2, 2)
    y = torch.tensor([[[1., 0.], [100., 100.]], [[0., 2.], [0., 2.]]])
    valid = torch.tensor([[True, False], [True, True]])
    result = reconstruction(x, y, valid, None, torch.zeros(2, dtype=torch.double))
    assert result['weighted_mse'] == 2.5
    assert result['explained_fraction'] == 0
    original = torch.randn(2, 3, 2)
    replaced = replace_valid(original, y, valid)
    torch.testing.assert_close(replaced[:, 0], original[:, 0])
    torch.testing.assert_close(replaced[0, 2], original[0, 2])
    torch.testing.assert_close(replaced[:, 1:][valid], y[valid])
