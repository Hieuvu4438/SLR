from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Support:
    selected_positions: tuple[int, ...]
    dense_indices: tuple[int, ...]
    weights: tuple[float, ...]
    interval: tuple[int, int]
    confidence: float
    word_index: int


def token_word_distributions(
    visual_tokens: torch.Tensor,
    word_tokens: torch.Tensor,
    *,
    tau_word: float = 0.07,
    tau_time: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    if visual_tokens.ndim != 2 or word_tokens.ndim != 2:
        raise ValueError("visual_tokens and word_tokens must be rank-2")
    if visual_tokens.shape[1] != word_tokens.shape[1] or not len(word_tokens):
        raise ValueError("token dimensions must match and at least one word is required")
    if tau_word <= 0 or tau_time <= 0:
        raise ValueError("temperatures must be positive")
    visual = F.normalize(visual_tokens.float(), dim=-1)
    words = F.normalize(word_tokens.float(), dim=-1)
    affinity = visual @ words.T
    p_word = torch.softmax(affinity / tau_word, dim=1)
    q_time = torch.softmax(affinity / tau_time, dim=0)
    return p_word, q_time


def _shortest_mass_interval(
    q: torch.Tensor, mass_min: float, min_tokens: int, max_tokens: int
) -> tuple[int, int] | None:
    best: tuple[int, float, int, int] | None = None
    prefix = torch.cat((q.new_zeros(1), torch.cumsum(q, 0)))
    for start in range(len(q)):
        for end in range(start + min_tokens, min(len(q), start + max_tokens) + 1):
            mass = float((prefix[end] - prefix[start]).item())
            if mass + 1e-12 < mass_min:
                continue
            candidate = (end - start, -mass, start, end)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else (best[2], best[3])


def select_support(
    p_word: torch.Tensor,
    q_time: torch.Tensor,
    dense_indices: torch.Tensor,
    word_index: int,
    *,
    mass_min: float = 0.60,
    min_support_tokens: int = 2,
    max_duration_fraction: float = 0.35,
    confidence_min: float = 0.45,
    second_mode_mass: float = 0.30,
) -> Support | None:
    if p_word.shape != q_time.shape:
        raise ValueError("p_word and q_time must share [F,K]")
    if dense_indices.ndim != 1 or dense_indices.shape[0] != p_word.shape[0]:
        raise ValueError("dense_indices must match the temporal axis")
    if not 0 <= word_index < p_word.shape[1]:
        raise IndexError(word_index)
    q = q_time[:, word_index]
    max_tokens = max(min_support_tokens, int(len(q) * max_duration_fraction))
    interval = _shortest_mass_interval(q, mass_min, min_support_tokens, max_tokens)
    if interval is None:
        return None
    start, end = interval
    if not bool((p_word[start:end].argmax(dim=1) == word_index).any()):
        return None
    support_mass = q[start:end].sum()
    weights = q[start:end] / support_mass
    confidence = float((weights * p_word[start:end, word_index]).sum().item())
    if confidence < confidence_min:
        return None
    # A substantial disconnected temporal mode makes the MVP pseudo-label ambiguous.
    if (
        float(q[:start].sum().item()) >= second_mode_mass
        or float(q[end:].sum().item()) >= second_mode_mass
    ):
        return None
    dense = dense_indices[start:end]
    if bool((dense < 0).any()):
        raise ValueError("support includes padding")
    full_dense_duration = int(dense_indices.max().item() - dense_indices.min().item() + 1)
    support_dense_duration = int(dense.max().item() - dense.min().item() + 1)
    if support_dense_duration / full_dense_duration > max_duration_fraction:
        return None
    return Support(
        selected_positions=tuple(range(start, end)),
        dense_indices=tuple(int(value) for value in dense.tolist()),
        weights=tuple(float(value) for value in weights.tolist()),
        interval=(int(dense.min().item()), int(dense.max().item()) + 1),
        confidence=confidence,
        word_index=word_index,
    )


def interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def reliability(
    support_a: Support,
    support_b: Support,
    *,
    occurrence_count: int,
    view_iou_min: float = 0.5,
) -> tuple[float, float]:
    iou = interval_iou(support_a.interval, support_b.interval)
    if iou < view_iou_min:
        return iou, 0.0
    rho = iou * support_a.confidence * min(1.0, occurrence_count / 5.0)
    return iou, min(1.0, max(0.0, rho))
