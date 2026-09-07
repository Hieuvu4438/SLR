from __future__ import annotations

import torch


def lexical_loss(
    z: torch.Tensor,
    weights: torch.Tensor,
    pos_e: torch.Tensor,
    neg_e: torch.Tensor,
    neg_valid: torch.Tensor,
    rho: torch.Tensor,
    *,
    margin: float = 0.1,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    if z.shape[:2] != weights.shape or z.shape[0] != pos_e.shape[0]:
        raise ValueError("incompatible lexical occurrence tensors")
    if neg_e.shape[:2] != neg_valid.shape or neg_e.shape[0] != z.shape[0]:
        raise ValueError("incompatible negative tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if z.shape[0] == 0:
        return 0.0 * z.sum(), torch.zeros((), dtype=torch.long, device=z.device)
    valid_occurrence = neg_valid.any(-1)
    if valid_occurrence.any():
        sums = weights[valid_occurrence].float().sum(-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
            raise ValueError("support weights must sum to one for eligible occurrences")
    pooled = (weights.detach().float().unsqueeze(-1) * z.float()).sum(1)
    positive = (pooled * pos_e.detach().float()).sum(-1)
    negative = torch.einsum("od,okd->ok", pooled, neg_e.detach().float())
    logits = (float(margin) - positive[:, None] + negative) / float(temperature)
    logits = logits.masked_fill(~neg_valid, float("-inf"))
    zeros = logits.new_zeros((logits.shape[0], 1))
    per_occurrence = torch.logsumexp(torch.cat((zeros, logits), dim=1), dim=1)
    numerator = (rho.detach().float() * per_occurrence * valid_occurrence).sum()
    return numerator, valid_occurrence.sum()


def globally_normalized_auxiliary(
    local_sum: torch.Tensor, local_count: torch.Tensor, *, world_size: int = 1
) -> torch.Tensor:
    count = local_count.detach().clone().to(local_sum.device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    if int(count.item()) == 0:
        return local_sum * 0.0
    return float(world_size) * local_sum / count.float()
