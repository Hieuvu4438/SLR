import torch
from methods.seds_adaptation.joint_exchange import JointExchange
from methods.seds_adaptation.joint_exchange import configure_joint_learning_rates


def test_identity_mask_gradients_and_roundtrip():
    torch.manual_seed(7)
    model = JointExchange(8, 4, (2, 2, 1))
    x = torch.randn(3, 5, 8)
    valid = torch.tensor([[1,1,1,1,1], [1,1,0,0,0], [0,0,0,0,0]], dtype=torch.bool)
    assert torch.equal(model(x, valid), x)
    model(x, valid).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.output.weight.add_(-.1*model.output.weight.grad)
    y = model(x, valid)
    altered = x.clone(); altered[~valid] = 1000
    assert torch.equal(model(altered, valid)[valid], y[valid])
    assert torch.equal(y[1:], x[1:])  # no legal cross-part keys / all absent
    model.zero_grad(); model(x, valid).square().mean().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    rebuilt = JointExchange(8, 4, (2, 2, 1))
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(rebuilt(x, valid), y)


def test_cross_vs_within_same_parameters_different_dependency():
    torch.manual_seed(8)
    cross = JointExchange(8, 4, (2, 2, 1), 'cross')
    within = JointExchange(8, 4, (2, 2, 1), 'within')
    assert sum(p.numel() for p in cross.parameters()) == sum(p.numel() for p in within.parameters())
    for a, b in zip(cross.parameters(), within.parameters()):
        with torch.no_grad(): b.copy_(a)
    with torch.no_grad():
        cross.output.weight.normal_(); within.output.weight.copy_(cross.output.weight)
    x = torch.randn(1, 5, 8); m = torch.ones(1, 5, dtype=torch.bool)
    other = x.clone(); other[:, 2:] *= -1
    assert not torch.allclose(cross(x,m)[:,:2], cross(other,m)[:,:2])
    assert torch.equal(within(x,m)[:,:2], within(other,m)[:,:2])


def test_optimizer_explicit_rates_preserve_groups_and_frozen_weights():
    model = torch.nn.Module()
    model.signbert = torch.nn.Module()
    model.signbert.joint_exchange = JointExchange(8,4,(2,2,1))
    model.fusion = torch.nn.Linear(8,8)
    model.backbone = torch.nn.Linear(8,8)
    model.backbone.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=.001)
    groups = configure_joint_learning_rates(optimizer,model,1e-4,1e-5)
    assert {(g['adaptation_group'],g['lr']) for g in groups} == {('joint',1e-4),('fusion',1e-5)}
    assert all(g['weight_decay'] == .001 for g in groups)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    for p in model.parameters():
        if p.requires_grad: p.grad = torch.ones_like(p)
    optimizer.step()
    for n,p in model.named_parameters():
        if n.startswith('backbone.'):
            assert torch.equal(p,before[n])
        else:
            lr = 1e-5 if n.startswith('fusion.') else 1e-4
            assert torch.allclose(p,before[n]-lr*(1+.001*before[n]),atol=1e-7,rtol=0)
