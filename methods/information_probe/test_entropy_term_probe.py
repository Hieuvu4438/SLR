import math
import torch
from .entropy_term_probe import inner_terms


def test_entropy_identity_and_derivative_against_autograd():
    a = torch.tensor([.9, .1, .2], dtype=torch.float64, requires_grad=True)
    e, h, u, derivative = inner_terms(a, 0)
    assert torch.allclose(u, e+.07*h-.07*math.log(3), atol=1e-12)
    grad = torch.autograd.grad(e, a, retain_graph=True)[0]
    assert torch.allclose(grad, derivative, atol=1e-12)
    assert (grad < 0).any() and torch.allclose(grad.sum(), torch.tensor(1., dtype=a.dtype))
    assert (torch.autograd.grad(u, a)[0] > 0).all()
