from __future__ import annotations

import torch
from torch.nn import functional as F


def _kl(teacher: torch.Tensor, student: torch.Tensor, temperature: float, dim: int) -> torch.Tensor:
    teacher_probability = F.softmax(teacher.detach().float() / temperature, dim=dim)
    student_log_probability = F.log_softmax(student.float() / temperature, dim=dim)
    return F.kl_div(student_log_probability, teacher_probability, reduction="batchmean")


def bidirectional_kl(
    teacher_scores: torch.Tensor, student_scores: torch.Tensor, *, temperature: float = 1.0
) -> torch.Tensor:
    if teacher_scores.shape != student_scores.shape or teacher_scores.ndim != 2:
        raise ValueError("teacher/student score matrices must match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    v2t = _kl(teacher_scores, student_scores, temperature, dim=1)
    t2v = _kl(teacher_scores.T, student_scores.T, temperature, dim=1)
    return 0.5 * (temperature**2) * (v2t + t2v)
