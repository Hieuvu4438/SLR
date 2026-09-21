"""Freeze an adapted native GCN without resetting fusion optimizer history."""


def freeze_gcn_keep_optimizer(model, optimizer):
    fusion = list(model.fusion.parameters())
    steps = [int(optimizer.state[p].get('step', 0)) for p in fusion]
    for parameter in model.signbert.embed.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    assert all(p.requires_grad for p in fusion)
    assert [int(optimizer.state[p].get('step', 0)) for p in fusion] == steps
    return dict(optimizer_reset=False, fusion_optimizer_steps=sorted(set(steps)))
