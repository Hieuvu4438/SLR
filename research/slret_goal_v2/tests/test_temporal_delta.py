import copy
import torch
from torch import nn
from methods.seds_adaptation.temporal_delta import TemporalDelta, attach_temporal_delta, configure_temporal_fusion


def test_identity_mask_invariance_gradient_and_order():
    torch.manual_seed(42)
    m=TemporalDelta(8,4)
    x=torch.randn(2,6,8)
    mask=torch.tensor([[0,0,0,0,1,1],[0,0,1,1,1,1]])
    assert torch.equal(m(x,mask),x)
    m(x,mask).square().mean().backward()
    assert m.output.weight.grad.abs().sum()>0
    with torch.no_grad():
        m.output.weight.add_(-.1*m.output.weight.grad)
    y=m(x,mask)
    assert torch.equal(y[:,0],x[:,0])
    assert torch.equal(y[mask.bool()],x[mask.bool()])
    assert torch.equal(y[1],x[1])  # singleton has no adjacent clip
    altered=x.clone(); altered[mask.bool()]=1000; altered[:,0]=-1000
    assert torch.equal(m(altered,mask)[0,1:4],y[0,1:4])
    reverse=x.clone(); reverse[0,1:4]=x[0,1:4].flip(0)
    assert not torch.allclose(m(reverse,mask)[0,1:4].flip(0),y[0,1:4])
    allpad=torch.ones_like(mask)
    assert torch.equal(m(x,allpad),x)


class Fake(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('clip','clip_rgb'):
            branch=nn.Module(); branch.visual=nn.Module()
            branch.visual.proj=nn.Parameter(torch.randn(8,8))
            setattr(self,name,branch)
        self.fusion=nn.Linear(8,8)
    def get_visual_output(self,x,mask):
        return mask,x@self.clip.visual.proj,x@self.clip_rgb.visual.proj


def test_attachment_frozen_backbone_and_checkpoint():
    model=Fake(); rebuilt=copy.deepcopy(model)
    x=torch.randn(2,5,8); mask=torch.zeros(2,5,dtype=torch.long)
    before=model.get_visual_output(x,mask)
    attach_temporal_delta(model,4)
    configure_temporal_fusion(model)
    after=model.get_visual_output(x,mask)
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    sum(model.fusion(y).square().mean() for y in after[1:]).backward()
    assert all(m.output.weight.grad.abs().sum()>0 for m in model.temporal_delta.values())
    assert model.clip.visual.proj.grad is None
    attach_temporal_delta(rebuilt,4)
    rebuilt.load_state_dict(model.state_dict(),strict=True)
    assert all(torch.equal(a,b) for a,b in zip(after,rebuilt.get_visual_output(x,mask)))
