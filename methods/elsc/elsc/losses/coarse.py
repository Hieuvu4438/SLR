from __future__ import annotations

import torch
from torch.nn import functional as F


def diagonal_cross_entropy(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("contrastive logits must be square")
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits.float(), targets)


def balanced_clcl_loss(
    i2t: torch.Tensor, t2i: torch.Tensor, *, dual_mix: float, mix_design: str = "balance"
) -> torch.Tensor:
    if i2t.shape != t2i.shape:
        raise ValueError("I2T and T2I shapes differ")
    eta = float(dual_mix)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("dual_mix must be in [0,1]")
    if mix_design == "balance":
        first = eta * diagonal_cross_entropy(i2t) + (1 - eta) * diagonal_cross_entropy(i2t.T)
        second = eta * diagonal_cross_entropy(t2i.T) + (1 - eta) * diagonal_cross_entropy(t2i)
    elif mix_design == "depart":
        first = eta * (diagonal_cross_entropy(i2t) + diagonal_cross_entropy(i2t.T))
        second = (1 - eta) * (diagonal_cross_entropy(t2i.T) + diagonal_cross_entropy(t2i))
    else:
        raise ValueError(f"unsupported mix_design: {mix_design}")
    return 0.5 * (first + second)
