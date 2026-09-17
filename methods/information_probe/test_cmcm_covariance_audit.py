import torch

from .cmcm_covariance_audit import finite_difference, reference


def test_reference_covariance_translation_and_observation_permutation():
    torch.manual_seed(5)
    x = torch.randn(1, 3, 2, 2, dtype=torch.float64)
    shifted = x+torch.tensor([1.,2.,3.], dtype=x.dtype)[None,:,None,None]
    permuted = x.flatten(2)[:, :, [2,0,3,1]].reshape_as(x)
    assert torch.allclose(reference(x), reference(shifted), atol=1e-14, rtol=0)
    assert torch.allclose(reference(x), reference(permuted), atol=1e-14, rtol=0)


def test_reference_autodiff_matches_composed_finite_difference():
    torch.manual_seed(6)
    x = (torch.randn(1,3,2,2,dtype=torch.float64)*.001).requires_grad_()
    g, = torch.autograd.grad(reference(x).sum(), x)
    fd = finite_difference(lambda z: reference(z).sum(), x.detach(), 1e-8)
    assert torch.allclose(g,fd,atol=1e-9,rtol=1e-8)
