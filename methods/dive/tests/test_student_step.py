from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dive.data.relations import build_pair_relations
from dive.models.evidence import EvidenceEncoder, clone_reference_and_student, state_hash
from dive.training.step import run_student_step


class WindowPose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, pose: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        features = self.projection(pose)
        shape = features.shape
        features = self.bn(features.reshape(-1, 4)).reshape(shape)
        result = []
        for batch, sample_grid in enumerate(grid):
            result.append(
                torch.stack([features[batch, left:right].mean(0) for left, right in sample_grid])
            )
        return torch.stack(result)


@dataclass(frozen=True)
class Contrast:
    sample_i: str
    sample_j: str
    unit_i: int
    unit_j: int
    q_i: tuple[float, ...] | None
    q_j: tuple[float, ...] | None
    h: float | None
    g: float


def _step_inputs():
    torch.manual_seed(41)
    pose = torch.randn(4, 5, 3, requires_grad=True)
    rgb = torch.randn(4, 2, 5, requires_grad=True)
    grid = torch.tensor([[[0, 3], [2, 5]]] * 4)
    video_mask = torch.ones(4, 2, dtype=torch.bool)
    text = torch.nn.functional.normalize(torch.randn(4, 2, 8), dim=-1).requires_grad_()
    text_mask = torch.ones(4, 2, dtype=torch.bool)
    baseline = torch.eye(4, requires_grad=True)
    samples = [f"s{index}" for index in range(4)]
    relations = build_pair_relations(
        [f"v{index}" for index in range(4)],
        [f"t{index}" for index in range(4)],
        {f"v{index}": {f"t{index}"} for index in range(4)},
    )
    return pose, rgb, grid, video_mask, text, text_mask, baseline, samples, relations


def test_student_step_filters_g_zero_before_nullable_support_and_freezes_inputs():
    torch.manual_seed(43)
    pair = clone_reference_and_student(
        EvidenceEncoder(WindowPose(), rgb_dim=5, pose_dim=4, hidden_dim=12, output_dim=8)
    )
    pose, rgb, grid, vm, text, tm, baseline, samples, relations = _step_inputs()
    with torch.no_grad():
        reference_u = pair.reference(pose, rgb, grid, vm).requires_grad_()
    reference_before = reference_u.detach().clone()
    student_before = state_hash(pair.student)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in pair.student.parameters() if parameter.requires_grad], lr=1e-3
    )
    sampled = [
        Contrast("s0", "s1", 0, 0, (0.7, 0.3), (0.4, 0.6), 1.2, 0.8),
        Contrast("s2", "s3", -1, -1, None, None, None, 0.0),
    ]
    metrics = run_student_step(
        pair.student,
        optimizer,
        pose=pose,
        rgb_local=rgb,
        grid=grid,
        video_mask=vm,
        text_units=text,
        text_mask=tm,
        baseline_scores=baseline,
        reference_evidence=reference_u,
        positives=relations.positives,
        candidates=relations.candidates,
        sample_to_video_row={"s0": 2, "s1": 0, "s2": 3, "s3": 1},
        sample_to_text_column={"s0": 1, "s1": 3, "s2": 0, "s3": 2},
        sampled_contrasts=sampled,
    )
    assert metrics.num_sampled == 2 and metrics.num_active == 1
    assert metrics.sum_reliability == 0.8
    assert state_hash(pair.student) != student_before
    assert text.grad is None and reference_u.grad is None and baseline.grad is None
    assert pose.grad is None and rgb.grad is None
    torch.testing.assert_close(reference_u.detach(), reference_before)


def test_student_step_with_h_zero_schedule_still_runs_global_backward():
    torch.manual_seed(47)
    pair = clone_reference_and_student(
        EvidenceEncoder(WindowPose(), rgb_dim=5, pose_dim=4, hidden_dim=12, output_dim=8)
    )
    pose, rgb, grid, vm, text, tm, baseline, samples, relations = _step_inputs()
    with torch.no_grad():
        reference_u = pair.reference(pose, rgb, grid, vm)
    optimizer = torch.optim.SGD(
        [parameter for parameter in pair.student.parameters() if parameter.requires_grad], lr=1e-3
    )
    metrics = run_student_step(
        pair.student,
        optimizer,
        pose=pose,
        rgb_local=rgb,
        grid=grid,
        video_mask=vm,
        text_units=text,
        text_mask=tm,
        baseline_scores=baseline,
        reference_evidence=reference_u,
        positives=relations.positives,
        candidates=relations.candidates,
        sample_to_video_row={sample: index for index, sample in enumerate(samples)},
        sample_to_text_column={sample: index for index, sample in enumerate(samples)},
        sampled_contrasts=[],
    )
    assert metrics.num_sampled == metrics.num_active == 0
    assert metrics.loss == metrics.retrieval_loss
