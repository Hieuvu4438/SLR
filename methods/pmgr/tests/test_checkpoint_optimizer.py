from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from pmgr.checkpoint import save_checkpoint, validate_resume
from pmgr.optimizer import build_inherited_bert_adam
from slr_common.utils import restore_rng_state


def _advance(model, optimizer):
    value = torch.randn(4)
    optimizer.zero_grad(set_to_none=True)
    loss = ((model(value) - 0.4) ** 2).sum()
    loss.backward()
    optimizer.step()


def test_exact_resume_matches_uninterrupted_rng_model_and_optimizer(tmp_path):
    torch.manual_seed(5)
    initial = nn.Linear(4, 1)
    uninterrupted = copy.deepcopy(initial)
    interrupted = copy.deepcopy(initial)
    optimizer_a = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01)
    optimizer_b = torch.optim.AdamW(interrupted.parameters(), lr=0.01)
    rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    for _ in range(3):
        _advance(uninterrupted, optimizer_a)
    torch.set_rng_state(rng)
    _advance(interrupted, optimizer_b)
    config = {"identity": "fixture"}
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path, model=interrupted, optimizer=optimizer_b, config=config, epoch=0,
        sampler_cursor=1, effective_step=1, best=None, provenance={"fixture": True},
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    validate_resume(raw, config)
    resumed = copy.deepcopy(initial)
    resumed.load_state_dict(raw["model"])
    optimizer_c = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    optimizer_c.load_state_dict(raw["optimizer"])
    restore_rng_state(raw["rng"])
    for _ in range(2):
        _advance(resumed, optimizer_c)
    for left, right in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    with pytest.raises(ValueError, match="config hash"):
        validate_resume(raw, {"identity": "changed"})


class _ClipFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = nn.Linear(1, 1, bias=False)


def test_inherited_bert_adam_initial_warmup_step_is_zero_then_updates():
    model = _ClipFixture()
    config = {
        "paths": {"cico_root": str(Path(__file__).resolve().parents[3] / "third_party/SLRT/CiCo/CLCL")},
        "training": {
            "learning_rate": 1e-3, "weight_decay": 0.001, "warmup_fraction": 0.1,
            "beta1": 0.9, "beta2": 0.98, "epsilon": 1e-6,
        },
    }
    optimizer = build_inherited_bert_adam(model, config, total_steps=20)
    before = model.clip.weight.detach().clone()
    model.clip.weight.grad = torch.ones_like(model.clip.weight)
    optimizer.step()
    torch.testing.assert_close(model.clip.weight, before, atol=0, rtol=0)
    model.clip.weight.grad = torch.ones_like(model.clip.weight)
    optimizer.step()
    assert not torch.equal(model.clip.weight, before)
