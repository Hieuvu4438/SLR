import copy
import torch

from methods.seds_adaptation.phase_modulation import SharedPhaseModulation
from methods.seds_adaptation.masked_pose import masked_learning_rates


def test_zero_identity_cls_and_padding():
    torch.manual_seed(3)
    module=SharedPhaseModulation(8,3,6)
    pose=torch.randn(2,7,8);rgb=torch.randn(2,7,8)
    mask=torch.tensor([[0,0,0,0,1,1,1],[0,0,0,0,0,0,1]])
    p,r=module(pose,rgb,mask)
    assert torch.equal(p,pose) and torch.equal(r,rgb)
    with torch.no_grad():module.scale['pose'].normal_();module.shift['rgb'].normal_()
    p,r=module(pose,rgb,mask)
    assert torch.equal(p[:,0],pose[:,0]) and torch.equal(r[:,0],rgb[:,0])
    assert torch.equal(p[0,4:],pose[0,4:]) and torch.equal(r[0,4:],rgb[0,4:])


def test_length_normalized_phase_and_stream_separation():
    module=SharedPhaseModulation(4,2,8)
    with torch.no_grad():module.shift['pose'][0].fill_(1.)
    x=torch.zeros(2,7,4);mask=torch.tensor([[0,0,0,0,1,1,1],[0,0,0,0,0,0,0]])
    pose,rgb=module(x,x,mask)
    assert torch.count_nonzero(rgb)==0
    # First and final valid clips use the same normalized phase endpoints.
    torch.testing.assert_close(pose[0,1],pose[1,1])
    torch.testing.assert_close(pose[0,3],pose[1,6])


def test_two_stage_gradient_and_state_roundtrip():
    torch.manual_seed(5)
    module=SharedPhaseModulation(4,2,6)
    x=torch.randn(2,5,4);mask=torch.zeros(2,5,dtype=torch.long)
    sum(t.sum() for t in module(x,x,mask)).backward()
    assert module.phase.grad.abs().sum()==0
    assert all(p.grad.abs().sum()>0 for values in (module.scale,module.shift) for p in values.values())
    with torch.no_grad():module.scale['pose'].add_(.01*module.scale['pose'].grad)
    module.zero_grad();module(x,x,mask)[0].sum().backward()
    assert module.phase.grad.abs().sum()>0
    state=copy.deepcopy(module.state_dict());rebuilt=SharedPhaseModulation(4,2,6)
    rebuilt.load_state_dict(state,strict=True)
    assert all(torch.equal(a,b) for a,b in zip(module(x,x,mask),rebuilt(x,x,mask)))


def test_registered_optimizer_groups():
    model=torch.nn.Module();model.fusion=torch.nn.Linear(4,4)
    model.signbert=torch.nn.Module();model.signbert.embed=torch.nn.Linear(4,4)
    model.masked_pose=torch.nn.Linear(4,4)
    model.temporal_modulation=SharedPhaseModulation(4,2,6)
    model.requires_grad_(False)
    model.fusion.requires_grad_(True);model.signbert.embed.requires_grad_(True)
    model.temporal_modulation.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.1)
    groups=masked_learning_rates(optimizer,model,control=True,
        temporal_modulation=True,temporal_modulation_lr=1e-4)
    assert {g['adaptation_group']:g['lr'] for g in groups} == {
        'encoder':1e-6,'fusion':1e-5,'temporal_modulation':1e-4}
