import torch

from methods.information_probe.gradient_probe import gradient_summary


def test_gradient_summary_mask_and_support():
    g=torch.tensor([[[3.,0.],[100.,100.]],[[1.,0.],[0.,0.]]])
    valid=torch.tensor([[True,False],[True,False]])
    duplicate=torch.tensor([True,False])
    aligned=gradient_summary(g,g,valid,duplicate)
    assert abs(aligned['global_cosine']-1)<1e-12
    assert abs(aligned['duplicate_squared_norm_mass_fraction']-.9)<1e-12
    opposed=gradient_summary(g,-g,valid,duplicate)
    assert abs(opposed['global_cosine']+1)<1e-12
    assert opposed['combined_gradient_norm']==0
