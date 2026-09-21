import copy
from types import SimpleNamespace

import torch
from torch import nn

from methods.seds_adaptation.graph_bottleneck_adapter import (
    GraphBottleneckAdapter, attach_graph_bottleneck_adapters,
    configure_graph_bottleneck_adapters, graph_adapter_learning_rates)


class FakeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.tcn = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(),
                                 nn.Conv2d(channels, channels, 1))

    def forward(self, value, adjacency):
        return value, adjacency


class FakeBackbone(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.st_gcn_networks = nn.ModuleList([FakeBlock(channels) for _ in range(3)])
        self.st_gcn_pool = nn.ModuleList([FakeBlock(channels) for _ in range(2)])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.signbert = nn.Module()
        self.signbert.embed = nn.Module()
        self.signbert.embed.st_gcn_hand = FakeBackbone()
        self.signbert.embed.st_gcn_body = FakeBackbone()
        self.fusion = nn.Linear(8, 8)


def test_zero_identity_then_two_stage_gradient():
    torch.manual_seed(4)
    module = GraphBottleneckAdapter(8, 3)
    value = torch.randn(2, 8, 1, 5)
    assert torch.equal(module(value), value)
    module(value).square().mean().backward()
    assert module.output.weight.grad.abs().sum() > 0
    assert module.input.weight.grad.abs().sum() == 0
    with torch.no_grad():
        module.output.weight.add_(-.01 * module.output.weight.grad)
    module.zero_grad(); module(value).square().mean().backward()
    assert module.input.weight.grad.abs().sum() > 0


def test_hooks_preserve_adjacency_and_roundtrip():
    torch.manual_seed(7)
    model = FakeModel()
    adapters = attach_graph_bottleneck_adapters(model, rank=3)
    assert len(adapters) == 10
    value = torch.randn(2, 8, 1, 4); adjacency = torch.eye(4)
    output, returned = model.signbert.embed.st_gcn_hand.st_gcn_networks[0](value, adjacency)
    assert torch.equal(output, value) and returned is adjacency
    with torch.no_grad(): adapters['hand_0'].output.weight.normal_()
    changed, returned = model.signbert.embed.st_gcn_hand.st_gcn_networks[0](value, adjacency)
    assert not torch.equal(changed, value) and returned is adjacency
    state = copy.deepcopy(model.state_dict())
    rebuilt = FakeModel(); attach_graph_bottleneck_adapters(rebuilt, rank=3)
    rebuilt.load_state_dict(state, strict=True)
    replay = rebuilt.signbert.embed.st_gcn_hand.st_gcn_networks[0](value, adjacency)[0]
    assert torch.equal(changed, replay)


def test_frozen_backbone_and_registered_optimizer_groups():
    model = FakeModel(); adapters = attach_graph_bottleneck_adapters(model, rank=3)
    model.requires_grad_(False); model.fusion.requires_grad_(True)
    active = [model.fusion]
    configure_graph_bottleneck_adapters(model, active)
    assert not any(p.requires_grad for p in model.signbert.embed.parameters())
    assert all(p.requires_grad for p in adapters.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    groups = graph_adapter_learning_rates(optimizer, model)
    assert {g['adaptation_group']: g['lr'] for g in groups} == {
        'fusion': 1e-5, 'graph_adapter': 1e-4}
