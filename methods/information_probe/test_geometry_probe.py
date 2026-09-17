import pytest
import torch

from methods.information_probe.geometry_probe import moments,fitted_transforms


def test_moments_equal_sequence_weight_and_ignore_padding():
    x=torch.tensor([[[1.,0.],[123.,456.]],[[0.,1.],[0.,1.]]])
    valid=torch.tensor([[True,False],[True,True]])
    mean,second=moments(x,valid)
    torch.testing.assert_close(mean,torch.tensor([.5,.5],dtype=torch.float64))
    torch.testing.assert_close(second,torch.eye(2,dtype=torch.float64)/2)
    with pytest.raises(ValueError):
        moments(x,torch.zeros_like(valid))


def test_geometry_controls_share_eigenspectrum_and_remove_one_dimension():
    mean=torch.zeros(4,dtype=torch.float64)
    second=torch.diag(torch.tensor([.1,.2,.3,.4],dtype=torch.float64))
    cov,values,transforms=fitted_transforms(mean,second,mean,second)
    torch.testing.assert_close(cov,second)
    for name in ('shared_pc1','random_pc1'):
        p=transforms[name][2]
        torch.testing.assert_close(p@p,p)
        torch.testing.assert_close(p.trace(),torch.tensor(3.,dtype=torch.float64))
    a,b=(transforms[k][2] for k in ('shared_whiten','random_whiten'))
    torch.testing.assert_close(torch.linalg.eigvalsh(a),torch.linalg.eigvalsh(b))
    regularized=.9*cov+.1*values.mean()*torch.eye(4,dtype=torch.float64)
    torch.testing.assert_close(a@regularized@a,torch.eye(4,dtype=torch.float64))
