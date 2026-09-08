"""Optimizer ownership and schedule contract tests for DIVE."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dive.models.evidence import EvidenceEncoder
from dive.training.optimizer import (
    OptimizerContractError,
    build_evidence_optimizer,
    build_warmup_cosine_scheduler,
)


class PoseWithNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.normalization = nn.LayerNorm(4)
        self.batch_norm = nn.BatchNorm1d(4)

    def forward(self, pose: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        projected = self.normalization(self.projection(pose[:, : grid.shape[1]]))
        shape = projected.shape
        return self.batch_norm(projected.reshape(-1, 4)).reshape(shape)


def _model() -> EvidenceEncoder:
    return EvidenceEncoder(PoseWithNorm(), rgb_dim=5, pose_dim=4, hidden_dim=12, output_dim=8)


def test_optimizer_assigns_every_trainable_parameter_once_with_exact_decay_policy():
    model = _model()
    bundle = build_evidence_optimizer(
        model,
        lr_projector=1e-4,
        lr_pose=1e-5,
        weight_decay=0.01,
    )
    optimized = [parameter for group in bundle.optimizer.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(item) for item in optimized} == {id(item) for item in expected}
    assert len(optimized) == len({id(item) for item in optimized})
    manifest = bundle.manifest
    groups = {(item["owner"], item["decay"]): item for item in manifest["groups"]}
    assert groups[("projector", True)]["lr"] == 1e-4
    assert groups[("pose", True)]["lr"] == 1e-5
    no_decay_names = {
        name
        for item in manifest["groups"]
        if not item["decay"]
        for name in item["parameters"]
    }
    assert "layer_norm.weight" in no_decay_names
    assert "projector_in.bias" in no_decay_names
    assert "pose_encoder.normalization.weight" in no_decay_names
    assert all("batch_norm" not in name for item in manifest["groups"] for name in item["parameters"])


def test_optimizer_rejects_a_forbidden_owned_parameter():
    model = _model()
    with pytest.raises(OptimizerContractError, match="forbidden frozen parameter"):
        build_evidence_optimizer(model, forbidden_modules=(model.pose_encoder,))


def test_linear_warmup_cosine_scheduler_reaches_configured_floor_and_resumes():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=10,
        warmup_fraction=0.2,
        minimum_lr_fraction=0.1,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    values = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-4)
    saved = scheduler.state_dict()
    restored_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=1e-3)
    restored = build_warmup_cosine_scheduler(
        restored_optimizer,
        total_steps=10,
        warmup_fraction=0.2,
        minimum_lr_fraction=0.1,
    )
    restored.load_state_dict(saved)
    assert restored.last_epoch == scheduler.last_epoch
