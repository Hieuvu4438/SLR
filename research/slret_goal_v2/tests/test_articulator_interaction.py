import sys
from pathlib import Path
import torch
from torch import nn
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.seds_adaptation.articulator_interaction import ArticulatorInteraction, attach_articulator_interaction


def test_identity_active_gradient_and_matched_capacity():
    torch.manual_seed(17)
    module = ArticulatorInteraction(8,4)
    control = ArticulatorInteraction(8,4,'additive')
    assert sum(p.numel() for p in module.parameters()) == sum(p.numel() for p in control.parameters())
    x = torch.randn(2,5,24)
    assert torch.equal(module(x),x)
    optimizer = torch.optim.SGD(module.parameters(),lr=.1)
    module(x).square().mean().backward()
    assert module.output.weight.grad.abs().sum() > 0
    optimizer.step(); optimizer.zero_grad()
    assert not torch.equal(module(x),x)
    module(x).square().mean().backward()
    assert module.left.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        module(torch.randn(2,5,25))


def test_attachment_and_checkpoint_reconstruction():
    class FakeSignBert(nn.Module):
        def __init__(self):
            super().__init__(); self.embed = nn.Linear(24,24)
        def gcn_emb(self,pose):
            return {'feat':self.embed(pose['raw'])}
    model = FakeSignBert()
    original_keys = set(model.state_dict())
    x = {'raw':torch.randn(2,5,24)}
    before = model.gcn_emb(x)['feat'].detach()
    attach_articulator_interaction(model,8,4)
    assert torch.equal(model.gcn_emb(x)['feat'],before)
    assert original_keys <= set(model.state_dict())
    rebuilt = FakeSignBert(); attach_articulator_interaction(rebuilt,8,4)
    rebuilt.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(rebuilt.gcn_emb(x)['feat'],model.gcn_emb(x)['feat'])
    with pytest.raises(ValueError):
        attach_articulator_interaction(model,8,4)


def test_frozen_tail_propagates_interaction_gradient_without_weight_drift():
    from methods.seds_adaptation.train_policies import configure_interaction_fusion
    model=nn.Module()
    model.signbert=nn.Module()
    model.signbert.articulator_interaction=ArticulatorInteraction(8,4)
    model.signbert.tail=nn.Sequential(nn.Linear(24,24),nn.BatchNorm1d(24))
    model.fusion=nn.Linear(24,1)
    active=configure_interaction_fusion(model)
    assert len(active)==2 and not model.signbert.tail.training
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    bn_before=model.signbert.tail[1].running_mean.clone()
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    x=torch.randn(2,5,24)
    for _ in range(2):
        y=model.signbert.articulator_interaction(x)
        loss=model.fusion(model.signbert.tail(y.reshape(-1,24))).square().mean()
        loss.backward()
        assert model.signbert.articulator_interaction.output.weight.grad.abs().sum()>0
        optimizer.step();optimizer.zero_grad()
    assert not torch.equal(model.signbert.articulator_interaction.output.weight,
                           torch.zeros_like(model.signbert.articulator_interaction.output.weight))
    assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in frozen.items())
    assert torch.equal(model.signbert.tail[1].running_mean,bn_before)
