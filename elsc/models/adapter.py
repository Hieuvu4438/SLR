from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalResidualAdapter(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or valid.shape != h.shape[:2]:
            raise ValueError("h must be [B,F,D] and valid must be [B,F]")
        delta = self.up(F.gelu(self.down(self.norm(h))))
        delta = delta * valid.unsqueeze(-1).to(delta.dtype)
        return h + delta


class LocalHead(nn.Module):
    def __init__(self, input_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, text_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, h_prime: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(h_prime).float(), dim=-1, eps=1e-6)
