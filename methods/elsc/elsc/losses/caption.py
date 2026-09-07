from __future__ import annotations

import torch


def matched_caption_loss(
    positive: torch.Tensor,
    negatives: torch.Tensor,
    negative_valid: torch.Tensor,
    rho: torch.Tensor,
    *,
    margin: float = 0.0,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    if negatives.shape != negative_valid.shape or negatives.shape[0] != positive.shape[0]:
        raise ValueError("caption score shapes are inconsistent")
    logits = (margin - positive.float()[:, None] + negatives.float()) / temperature
    logits = logits.masked_fill(~negative_valid, float("-inf"))
    per_occurrence = torch.logsumexp(
        torch.cat((logits.new_zeros((logits.shape[0], 1)), logits), dim=1), dim=1
    )
    eligible = negative_valid.any(-1)
    return (rho.detach().float() * per_occurrence * eligible).sum(), eligible.sum()
