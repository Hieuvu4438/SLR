import sys
from pathlib import Path
import torch
from torch.nn import functional as F
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.seds_adaptation.smoothed_contrastive import SmoothedCrossEn


def test_native_zero_parity_and_torch_reference_gradient():
    scores=torch.randn(5,5,requires_grad=True)
    native=-torch.diag(F.log_softmax(scores,dim=-1)).mean()
    assert torch.equal(SmoothedCrossEn(0)(scores),native)
    actual=SmoothedCrossEn(.05)(scores)
    reference=F.cross_entropy(scores,torch.arange(5),label_smoothing=.05)
    assert torch.allclose(actual,reference)
    a=torch.autograd.grad(actual,scores,retain_graph=True)[0]
    b=torch.autograd.grad(reference,scores)[0]
    assert torch.allclose(a,b,atol=1e-7)
    with pytest.raises(ValueError):
        SmoothedCrossEn(.05)(torch.randn(3,5))
