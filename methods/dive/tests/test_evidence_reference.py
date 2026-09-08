from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dive.data.relations import build_pair_relations
from dive.losses import retrieval_loss
from dive.models.evidence import (
    EvidenceEncoder,
    assert_equal_independent_state,
    clone_reference_and_student,
    state_hash,
)
from dive.models.scoring import compose_score, evidence_score_block
from dive.training.selection import DevCandidate, select_earliest_best


class TinyWindowPoseEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 4) -> None:
        super().__init__()
        self.frame_projection = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, pose: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        frame_features = self.frame_projection(pose)
        shape = frame_features.shape
        normalized = self.batch_norm(frame_features.reshape(-1, shape[-1])).reshape(shape)
        clips: list[torch.Tensor] = []
        for batch_index in range(len(pose)):
            sample: list[torch.Tensor] = []
            for left, right in grid[batch_index].tolist():
                sample.append(normalized[batch_index, left:right].mean(dim=0))
            clips.append(torch.stack(sample))
        return torch.stack(clips)


def _inputs():
    generator = torch.Generator().manual_seed(29)
    pose = torch.randn(2, 6, 3, generator=generator)
    rgb = torch.randn(2, 2, 5, generator=generator)
    grid = torch.tensor([[[0, 3], [2, 6]], [[0, 2], [3, 6]]], dtype=torch.int64)
    valid = torch.ones(2, 2, dtype=torch.bool)
    return pose, rgb, grid, valid


def _model() -> EvidenceEncoder:
    torch.manual_seed(31)
    return EvidenceEncoder(
        TinyWindowPoseEncoder(), rgb_dim=5, pose_dim=4, hidden_dim=12, output_dim=8
    )


def test_bn_affine_and_statistics_remain_frozen_after_train_and_step():
    model = _model()
    model.train()
    bn = model.pose_encoder.batch_norm
    assert bn.training is False
    assert bn.weight.requires_grad is False and bn.bias.requires_grad is False
    before_mean = bn.running_mean.clone()
    before_variance = bn.running_var.clone()
    pose, rgb, grid, valid = _inputs()
    optimizer = torch.optim.AdamW([item for item in model.parameters() if item.requires_grad], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    model(pose, rgb, grid, valid).sum().backward()
    optimizer.step()
    torch.testing.assert_close(bn.running_mean, before_mean)
    torch.testing.assert_close(bn.running_var, before_variance)


def test_reference_student_are_equal_without_shared_storage_then_diverge_safely():
    pair = clone_reference_and_student(_model())
    assert state_hash(pair.reference) == state_hash(pair.student) == pair.initial_state_hash
    assert_equal_independent_state(pair.reference, pair.student)
    reference_before = copy.deepcopy(pair.reference.state_dict())
    pose, rgb, grid, valid = _inputs()
    text = torch.nn.functional.normalize(torch.randn(2, 2, 8), dim=-1)
    text_mask = torch.ones(2, 2, dtype=torch.bool)
    with torch.no_grad():
        reference_u = pair.reference(pose, rgb, grid, valid)
        reference_score, pair_valid = evidence_score_block(reference_u, text, valid, text_mask)
    student_u = pair.student(pose, rgb, grid, valid)
    student_score, _ = evidence_score_block(student_u, text, valid, text_mask)
    baseline = torch.tensor([[0.2, -0.1], [-0.2, 0.3]])
    composed = compose_score(baseline, student_score, reference_score, pair_valid, gamma=0.1)
    torch.testing.assert_close(composed, baseline, atol=1e-6, rtol=1e-5)
    relations = build_pair_relations(
        ["v0", "v1"], ["t0", "t1"], {"v0": {"t0"}, "v1": {"t1"}}
    )
    loss = retrieval_loss(composed, relations.positives, relations.candidates)
    loss.backward()
    gradients = [item.grad for item in pair.student.parameters() if item.requires_grad]
    assert any(item is not None and bool((item != 0).any()) for item in gradients)
    optimizer = torch.optim.SGD(
        [item for item in pair.student.parameters() if item.requires_grad], lr=1e-2
    )
    optimizer.step()
    assert state_hash(pair.student) != pair.initial_state_hash
    for name, value in pair.reference.state_dict().items():
        torch.testing.assert_close(value, reference_before[name])


def test_raw_frames_outside_clip_window_have_zero_influence_and_gradient():
    model = _model().eval()
    pose, rgb, grid, valid = _inputs()
    original = model(pose, rgb, grid, valid)
    changed = pose.clone()
    changed[0, 3:] += 100.0
    perturbed = model(changed, rgb, grid, valid)
    torch.testing.assert_close(perturbed[0, 0], original[0, 0])
    differentiable = pose.clone().requires_grad_(True)
    model(differentiable, rgb, grid, valid)[0, 0].sum().backward()
    assert differentiable.grad is not None
    assert torch.equal(differentiable.grad[0, 3:], torch.zeros_like(differentiable.grad[0, 3:]))


def test_warmup_selection_uses_dev_endpoint_and_earliest_tie():
    candidates = [
        DevCandidate(0, 0, 0.4, 0.6, "initial.pt"),
        DevCandidate(1, 10, 0.5, 0.7, "epoch1.pt"),
        DevCandidate(2, 20, 0.6, 0.6, "epoch2.pt"),
    ]
    assert select_earliest_best(candidates, split="dev").checkpoint_path == "epoch1.pt"
    with pytest.raises(ValueError, match="dev-only"):
        select_earliest_best(candidates, split="test")
