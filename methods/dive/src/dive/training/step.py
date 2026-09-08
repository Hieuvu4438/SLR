from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import torch
from torch import Tensor

from dive.losses import local_loss_active, pair_loss_active, retrieval_loss
from dive.models.evidence import EvidenceEncoder
from dive.models.scoring import compose_score, evidence_score_block


class ContrastLike(Protocol):
    sample_i: str
    sample_j: str
    unit_i: int
    unit_j: int
    q_i: Sequence[float] | None
    q_j: Sequence[float] | None
    h: float | None
    g: float


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    retrieval_loss: float
    pair_loss: float
    local_loss: float
    num_sampled: int
    num_active: int
    sum_reliability: float
    active_feasibility: int
    gradient_norm: float


def _quartets(matrix: Tensor, endpoint_rows: Tensor, endpoint_columns: Tensor) -> Tensor:
    output: list[Tensor] = []
    for rows, columns in zip(endpoint_rows, endpoint_columns, strict=True):
        output.append(matrix[rows[:, None], columns[None, :]])
    return torch.stack(output) if output else matrix.new_empty((0, 2, 2))


def run_student_step(
    student: EvidenceEncoder,
    optimizer: torch.optim.Optimizer,
    *,
    pose: Tensor,
    rgb_local: Tensor,
    grid: Tensor,
    video_mask: Tensor,
    text_units: Tensor,
    text_mask: Tensor,
    baseline_scores: Tensor,
    reference_evidence: Tensor,
    positives: Tensor,
    candidates: Tensor,
    sample_to_video_row: Mapping[str, int],
    sample_to_text_column: Mapping[str, int],
    sampled_contrasts: Sequence[ContrastLike],
    gamma_train: float = 0.1,
    tau_alignment: float = 0.07,
    tau_retrieval: float = 0.07,
    tau_pair: float = 0.07,
    tau_local_margin: float = 0.1,
    local_margin_alpha: float = 0.25,
    lambda_pair: float = 0.1,
    lambda_local: float = 0.1,
    grad_clip_norm: float = 1.0,
) -> StepMetrics:
    if grad_clip_norm <= 0 or lambda_pair < 0 or lambda_local < 0:
        raise ValueError("gradient clip and loss weights are invalid")
    student.train()
    optimizer.zero_grad(set_to_none=True)
    frozen_text = text_units.detach()
    frozen_reference = reference_evidence.detach()
    frozen_baseline = baseline_scores.detach()
    student_u = student(pose.detach(), rgb_local.detach(), grid, video_mask)
    student_scores, pair_valid = evidence_score_block(
        student_u, frozen_text, video_mask, text_mask, tau_alignment
    )
    reference_scores, reference_valid = evidence_score_block(
        frozen_reference, frozen_text, video_mask, text_mask, tau_alignment
    )
    if not torch.equal(pair_valid, reference_valid):
        raise ValueError("student/reference pair validity differs")
    scores = compose_score(
        frozen_baseline, student_scores, reference_scores, pair_valid, gamma_train
    )
    loss_retrieval = retrieval_loss(scores, positives, candidates, tau_retrieval)
    graph_zero = student_u.sum() * 0.0
    loss_local = graph_zero
    loss_pair = scores.sum() * 0.0
    feasibility = torch.empty((0, 4), dtype=torch.bool, device=scores.device)
    active = [record for record in sampled_contrasts if record.g > 0]
    if active:
        rows: list[tuple[int, int]] = []
        columns: list[tuple[int, int]] = []
        q_values: list[Tensor] = []
        target_values: list[Tensor] = []
        reliability: list[float] = []
        distances: list[float] = []
        for record in active:
            if record.q_i is None or record.q_j is None or record.h is None:
                raise ValueError("active contrast requires q and text distance")
            if record.sample_i not in sample_to_video_row or record.sample_j not in sample_to_video_row:
                raise ValueError("active contrast video endpoint is absent from batch")
            if record.sample_i not in sample_to_text_column or record.sample_j not in sample_to_text_column:
                raise ValueError("active contrast text endpoint is absent from batch")
            video_rows = (
                sample_to_video_row[record.sample_i],
                sample_to_video_row[record.sample_j],
            )
            text_columns = (
                sample_to_text_column[record.sample_i],
                sample_to_text_column[record.sample_j],
            )
            if record.unit_i < 0 or record.unit_j < 0:
                raise ValueError("active target unit index cannot be negative")
            if not text_mask[text_columns[0], record.unit_i] or not text_mask[
                text_columns[1], record.unit_j
            ]:
                raise ValueError("active target unit is invalid or truncated")
            rows.append(video_rows)
            columns.append(text_columns)
            q_values.append(
                torch.as_tensor(
                    (record.q_i, record.q_j), device=student_u.device, dtype=student_u.dtype
                )
            )
            target_values.append(
                torch.stack(
                    (
                        frozen_text[text_columns[0], record.unit_i],
                        frozen_text[text_columns[1], record.unit_j],
                    )
                )
            )
            reliability.append(float(record.g))
            distances.append(float(record.h))
        row_indices = torch.as_tensor(rows, dtype=torch.int64, device=scores.device)
        column_indices = torch.as_tensor(columns, dtype=torch.int64, device=scores.device)
        u_pair = student_u[row_indices]
        q_pair = torch.stack(q_values)
        d_pair = torch.stack(target_values)
        g_active = torch.as_tensor(reliability, device=scores.device, dtype=scores.dtype)
        text_distance = torch.as_tensor(distances, device=scores.device, dtype=scores.dtype)
        if lambda_local > 0:
            loss_local, _, _ = local_loss_active(
                u_pair,
                q_pair,
                d_pair,
                g_active,
                text_distance,
                len(sampled_contrasts),
                tau_delta=tau_local_margin,
                alpha_margin=local_margin_alpha,
            )
        if lambda_pair > 0:
            full_quartets = _quartets(scores, row_indices, column_indices)
            baseline_quartets = _quartets(frozen_baseline, row_indices, column_indices)
            loss_pair, feasibility = pair_loss_active(
                full_quartets,
                baseline_quartets,
                g_active,
                len(sampled_contrasts),
                gamma_train=gamma_train,
                tau_pair=tau_pair,
            )
    loss = loss_retrieval + lambda_pair * loss_pair + lambda_local * loss_local
    if not torch.isfinite(loss):
        raise FloatingPointError("student loss is nonfinite")
    loss.backward()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not any(parameter.grad is not None for parameter in trainable):
        raise RuntimeError("student received no gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    optimizer.step()
    return StepMetrics(
        loss=float(loss.detach()),
        retrieval_loss=float(loss_retrieval.detach()),
        pair_loss=float(loss_pair.detach()),
        local_loss=float(loss_local.detach()),
        num_sampled=len(sampled_contrasts),
        num_active=len(active),
        sum_reliability=sum(float(record.g) for record in active),
        active_feasibility=int(feasibility.sum()),
        gradient_norm=float(gradient_norm),
    )
