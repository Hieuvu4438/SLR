from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from pmgr.checkpoint import save_checkpoint, validate_resume
from pmgr.optimizer import build_inherited_bert_adam
from pmgr.provenance import build_resume_invariants, implementation_identity
from slr_common.utils import restore_rng_state


ROOT = Path(__file__).resolve().parents[3]


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
        sampler_state={"epoch": 0, "cursor": 1, "permutation": [1, 0]},
        progress={"effective_steps": 1, "groups_processed": 2},
        validation_history=[],
        resume_invariants={"fixture": "unchanged"},
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    validate_resume(raw, config, resume_invariants={"fixture": "unchanged"})
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
    with pytest.raises(ValueError, match="resume invariant changed"):
        validate_resume(raw, config, resume_invariants={"fixture": "changed"})


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


def test_resume_invariants_fingerprint_code_dependencies_and_all_resources(tmp_path):
    paths = {}
    for name in (
        "train_manifest",
        "validation_manifest",
        "train_index",
        "validation_index",
        "baseline_resolved_args",
        "initialization_checkpoint",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = str(path)
    paths["cico_root"] = str(ROOT / "third_party/SLRT/CiCo/CLCL")
    config = {
        "upstream_commit": "38a4f7b00da7a858d59b7fabe5093876a84db8e0",
        "paths": paths,
        "engine": {"precision": "fp32"},
    }
    invariants = build_resume_invariants(config, repository_root=ROOT, world_size=1)
    assert invariants["implementation"] == implementation_identity(ROOT)
    assert set(invariants["resources"]) == {
        "train_manifest",
        "validation_manifest",
        "train_index",
        "validation_index",
        "baseline_resolved_args",
        "initialization_checkpoint",
    }
    assert invariants["dependencies"]["packages"]["torch"] == torch.__version__
    assert all(record["sha256"] for record in invariants["resources"].values())
