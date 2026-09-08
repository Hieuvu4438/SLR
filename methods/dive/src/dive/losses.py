from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def margins4(quartet: Tensor) -> Tensor:
    if quartet.shape[-2:] != (2, 2):
        raise ValueError("quartet must end with shape [2,2]")
    return torch.stack(
        (
            quartet[..., 0, 0] - quartet[..., 0, 1],
            quartet[..., 1, 1] - quartet[..., 1, 0],
            quartet[..., 0, 0] - quartet[..., 1, 0],
            quartet[..., 1, 1] - quartet[..., 0, 1],
        ),
        dim=-1,
    )


def smooth_margin(delta: Tensor, margin: Tensor | float, tau: float) -> Tensor:
    if tau <= 0:
        raise ValueError("margin temperature must be positive")
    return tau * F.softplus((torch.as_tensor(margin, device=delta.device, dtype=delta.dtype) - delta) / tau)


def directional_retrieval(
    scores: Tensor,
    positives: Tensor,
    candidates: Tensor,
    tau: float = 0.07,
) -> Tensor:
    if scores.shape != positives.shape or scores.shape != candidates.shape:
        raise ValueError("scores, positives, and candidates must have identical shapes")
    if positives.dtype != torch.bool or candidates.dtype != torch.bool:
        raise TypeError("positives/candidates must be bool with True=eligible")
    if tau <= 0:
        raise ValueError("retrieval temperature must be positive")
    if bool((positives & ~candidates).any()):
        raise ValueError("positives must be a subset of candidates")
    if not bool(positives.any(dim=1).all()):
        raise ValueError("every query must have at least one positive")
    logits = scores / tau
    numerator = torch.logsumexp(logits.masked_fill(~positives, float("-inf")), dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~candidates, float("-inf")), dim=1)
    return (denominator - numerator).mean()


def retrieval_loss(
    scores: Tensor,
    positives: Tensor,
    candidates: Tensor,
    tau: float = 0.07,
) -> Tensor:
    return 0.5 * (
        directional_retrieval(scores, positives, candidates, tau)
        + directional_retrieval(scores.T, positives.T, candidates.T, tau)
    )


def local_loss_active(
    u_pair: Tensor,
    q_pair: Tensor,
    d_pair: Tensor,
    reliability: Tensor,
    text_distance: Tensor,
    num_sampled_contrasts: int,
    *,
    tau_delta: float = 0.1,
    alpha_margin: float = 0.25,
) -> tuple[Tensor, Tensor, Tensor]:
    active = u_pair.shape[0]
    if u_pair.ndim != 4 or u_pair.shape[1] != 2:
        raise ValueError("u_pair must have shape [H_active,2,Nv,D]")
    if q_pair.shape != u_pair.shape[:3]:
        raise ValueError("q_pair must have shape [H_active,2,Nv]")
    if d_pair.shape != (active, 2, u_pair.shape[-1]):
        raise ValueError("d_pair must have shape [H_active,2,D]")
    if reliability.shape != (active,) or text_distance.shape != (active,):
        raise ValueError("reliability and text_distance must have shape [H_active]")
    if num_sampled_contrasts < active:
        raise ValueError("sampled contrast count cannot be smaller than active count")
    if active == 0:
        empty = u_pair.new_empty((0, 2, 2))
        return u_pair.sum() * 0.0, empty, u_pair.new_empty((0, 4))
    if bool((reliability <= 0).any()):
        raise ValueError("active reliability values must be positive")
    if not torch.allclose(q_pair.sum(-1), torch.ones_like(q_pair.sum(-1)), atol=1e-6, rtol=1e-5):
        raise ValueError("each active q distribution must sum to one")
    quartet = torch.einsum("han,hand,hbd->hab", q_pair, u_pair, d_pair)
    delta = margins4(quartet)
    target = alpha_margin * text_distance[:, None]
    per_pair = smooth_margin(delta, target, tau_delta).mean(dim=-1)
    loss = (reliability * per_pair).sum() / max(1, num_sampled_contrasts)
    return loss, quartet, delta


def pair_loss_active(
    full_quartet: Tensor,
    baseline_quartet: Tensor,
    reliability: Tensor,
    num_sampled_contrasts: int,
    *,
    gamma_train: float = 0.1,
    tau_pair: float = 0.07,
    target_margin: float = 0.0,
) -> tuple[Tensor, Tensor]:
    active = full_quartet.shape[0]
    if full_quartet.shape != baseline_quartet.shape or full_quartet.shape[-2:] != (2, 2):
        raise ValueError("full and baseline quartets must share shape [H_active,2,2]")
    if reliability.shape != (active,):
        raise ValueError("reliability must have shape [H_active]")
    if num_sampled_contrasts < active:
        raise ValueError("sampled contrast count cannot be smaller than active count")
    if gamma_train < 0:
        raise ValueError("gamma_train must be nonnegative")
    if active == 0:
        return full_quartet.sum() * 0.0, torch.empty((0, 4), dtype=torch.bool, device=full_quartet.device)
    with torch.no_grad():
        feasible = margins4(baseline_quartet) >= -2.0 * gamma_train
    delta = margins4(full_quartet)
    per_comparison = smooth_margin(delta, target_margin, tau_pair)
    per_pair = (feasible.to(delta.dtype) * per_comparison).mean(dim=-1)
    loss = (reliability * per_pair).sum() / max(1, num_sampled_contrasts)
    return loss, feasible


def one_sided_preservation(
    current_margins: Tensor,
    baseline_margins: Tensor,
    tolerance: float,
) -> Tensor:
    if current_margins.shape != baseline_margins.shape:
        raise ValueError("current and baseline margins must share shape")
    if tolerance < 0:
        raise ValueError("preservation tolerance must be nonnegative")
    if current_margins.numel() == 0:
        return current_margins.sum() * 0.0
    return F.relu(baseline_margins.detach() - tolerance - current_margins).square().mean()
