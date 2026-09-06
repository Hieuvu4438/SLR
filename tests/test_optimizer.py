from __future__ import annotations

import torch
from torch import nn

from elsc.train import _optimizer


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(1.0))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()
        self.adapter = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
        self.local_head = nn.Linear(4, 3, bias=False)
        self.adapter_enabled = True


def test_optimizer_uses_explicit_cico_moments_and_no_decay_for_1d():
    model = _Model()
    config = {
        "method": "elsc",
        "model": {
            "backbone_frozen": False,
            "logit_scale_frozen": True,
            "adapter": {"trainable": True},
        },
        "train": {
            "core_lr": 1e-5,
            "adapter_lr": 1e-4,
            "head_lr": 2e-4,
            "weight_decay": 1e-3,
            "beta1": 0.9,
            "beta2": 0.98,
            "epsilon": 1e-6,
            "exclude_bias_and_1d_from_weight_decay": True,
        },
    }
    optimizer = _optimizer(model, config)
    assert optimizer.defaults["betas"] == (0.9, 0.98)
    assert optimizer.defaults["eps"] == 1e-6
    by_name = {group["name"]: group for group in optimizer.param_groups}
    assert by_name["core_decay"]["weight_decay"] == 1e-3
    assert by_name["core_no_decay"]["weight_decay"] == 0.0
    assert by_name["adapter_no_decay"]["weight_decay"] == 0.0
    assert by_name["local_head_decay"]["lr"] == 2e-4
    assert model.core.clip.logit_scale.requires_grad is False
