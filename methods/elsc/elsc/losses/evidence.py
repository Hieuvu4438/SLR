from __future__ import annotations

import torch

from slr_common.utils import stable_seed


def apply_input_intervention(
    h: torch.Tensor, remove: torch.Tensor, valid: torch.Tensor, fill: torch.Tensor | float
) -> torch.Tensor:
    if remove.shape != valid.shape or h.shape[:2] != valid.shape:
        raise ValueError("h/remove/valid shapes are inconsistent")
    if bool((remove & ~valid).any()):
        raise ValueError("cannot intervene on padding")
    fill_tensor = torch.as_tensor(fill, dtype=h.dtype, device=h.device)
    if fill_tensor.ndim == 0:
        fill_tensor = fill_tensor.expand(h.shape[-1])
    if fill_tensor.shape != (h.shape[-1],):
        raise ValueError("fill must be scalar or [D]")
    return torch.where(remove.unsqueeze(-1), fill_tensor.view(1, 1, -1), h)


def huber(value: torch.Tensor, delta: float) -> torch.Tensor:
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    absolute = value.abs()
    return torch.where(absolute <= delta, 0.5 * value.square(), delta * (absolute - 0.5 * delta))


def evidence_losses(
    delta_clean: torch.Tensor,
    delta_evidence_removed: torch.Tensor,
    delta_control_removed: torch.Tensor,
    rho: torch.Tensor,
    *,
    dependence_margin: float = 0.02,
    huber_delta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = (delta_clean, delta_evidence_removed, delta_control_removed, rho)
    if any(tensor.shape != delta_clean.shape for tensor in tensors):
        raise ValueError("evidence tensors must have the same shape")
    if delta_clean.numel() == 0:
        zero = delta_clean.sum() * 0.0
        return zero, zero
    clean = delta_clean.float()
    dep_change = clean - delta_evidence_removed.float()
    control_change = clean - delta_control_removed.float()
    reliability = rho.detach().float()
    dependence = (reliability * torch.relu(float(dependence_margin) - dep_change)).mean()
    invariance = (reliability * huber(control_change, huber_delta)).mean()
    return dependence, invariance


def receptive_field_closure(
    rf_start: torch.Tensor, rf_end: torch.Tensor, interval: tuple[float, float], valid: torch.Tensor
) -> torch.Tensor:
    if rf_start.shape != rf_end.shape or rf_start.shape != valid.shape:
        raise ValueError("RF metadata shapes differ")
    start, end = interval
    if not start < end:
        raise ValueError("support interval must be non-empty and half-open")
    return valid & (rf_start < end) & (rf_end > start)


def select_matched_control(
    rf_start: torch.Tensor,
    rf_end: torch.Tensor,
    valid: torch.Tensor,
    evidence_mask: torch.Tensor,
    forbidden_mask: torch.Tensor,
    *,
    seed: int,
    pair_id: str,
    target_id: int,
    duration_tolerance: float = 0.10,
) -> torch.Tensor | None:
    if not (
        rf_start.shape == rf_end.shape == valid.shape == evidence_mask.shape == forbidden_mask.shape
    ):
        raise ValueError("control-selection tensors must share shape [F]")
    evidence_positions = evidence_mask.nonzero(as_tuple=False).flatten()
    if not len(evidence_positions):
        return None
    evidence_start = float(rf_start[evidence_positions].min())
    evidence_end = float(rf_end[evidence_positions].max())
    evidence_duration = evidence_end - evidence_start
    evidence_count = int(evidence_mask.sum())
    candidates: list[torch.Tensor] = []
    valid_positions = valid.nonzero(as_tuple=False).flatten().tolist()
    for left in range(len(valid_positions)):
        for right in range(left + 1, len(valid_positions) + 1):
            positions = valid_positions[left:right]
            interval_start = float(rf_start[positions].min())
            interval_end = float(rf_end[positions].max())
            mask = receptive_field_closure(rf_start, rf_end, (interval_start, interval_end), valid)
            if int(mask.sum()) != evidence_count:
                continue
            duration = interval_end - interval_start
            if (
                evidence_duration <= 0
                or abs(duration - evidence_duration) / evidence_duration > duration_tolerance
            ):
                continue
            if bool((mask & evidence_mask).any()) or bool((mask & forbidden_mask).any()):
                continue
            candidates.append(mask)
    if not candidates:
        return None
    index = stable_seed(seed, pair_id, target_id, "matched_control") % len(candidates)
    return candidates[index]
