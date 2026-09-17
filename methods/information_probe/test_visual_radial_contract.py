"""Independent checks of the final-layer feasibility algebra, not retrieval."""
import torch
from torch.nn import functional as F


def test_minimum_norm_feasibility_and_equal_variance_construction():
    gen = torch.Generator().manual_seed(42)
    dim, out = 8, 3
    p = torch.randn(dim, out, dtype=torch.float64, generator=gen)
    g = torch.linspace(.5, 1.5, dim, dtype=torch.float64)
    beta = .01 * g
    matrix = torch.cat(((g[:, None]*p).T, torch.ones(1, dim).double()/dim**.5))
    inverse = torch.linalg.pinv(matrix)
    null = torch.randn(dim, dtype=torch.float64, generator=gen)
    null -= inverse @ (matrix @ null)
    null /= null.norm()
    target = torch.tensor([.1, .2, .3], dtype=torch.float64)
    outputs, variances = [], []
    for scale in (.9, 1.1):
        rhs = torch.cat((scale*target - beta@p, torch.zeros(1)))
        minimum = inverse @ rhs
        assert minimum.square().sum() < .9*dim
        torch.testing.assert_close(matrix @ minimum, rhs, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(minimum @ null, torch.tensor(0.).double(), atol=1e-12, rtol=0)
        u = minimum + (.9*dim-minimum.square().sum()).sqrt()*null
        h = u * (1e-5/.1)**.5
        outputs.append(F.layer_norm(h, (dim,), g, beta, 1e-5) @ p)
        variances.append(h.var(unbiased=False))
    torch.testing.assert_close(variances[0], variances[1], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(outputs[0], .9*target, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(outputs[1], 1.1*target, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(F.normalize(outputs[0], dim=0),
                               F.normalize(outputs[1], dim=0), atol=1e-12, rtol=1e-12)


def test_minimum_norm_outside_ball_cannot_be_repaired_by_nullspace():
    matrix = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    minimum = torch.linalg.pinv(matrix) @ torch.tensor([2., 0.], dtype=torch.float64)
    null = torch.tensor([0., 0., 1.], dtype=torch.float64)
    assert minimum.square().sum() > 3
    for weight in (-2., -1., 0., 1., 2.):
        assert (minimum + weight*null).square().sum() >= minimum.square().sum()
