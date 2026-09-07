from __future__ import annotations

from typing import Any

import torch
from torch import nn

from elsc.models.adapter import LocalHead, LocalResidualAdapter
from slr_common.upstream.cico_bridge import CiCoBridge, TextEncoding, VideoEncoding


class ELSCRetriever(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        *,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        text_dim: int = 512,
        core_frozen: bool = True,
        adapter_enabled: bool = True,
    ):
        super().__init__()
        self.core = core
        self.adapter_enabled = adapter_enabled
        self.adapter = LocalResidualAdapter(input_dim, hidden_dim)
        self.local_head = LocalHead(input_dim, text_dim)
        self.bridge = CiCoBridge(core)
        self.core_frozen = core_frozen
        if core_frozen:
            self.core.requires_grad_(False)
            self.core.eval()

    def train(self, mode: bool = True) -> "ELSCRetriever":
        super().train(mode)
        if self.core_frozen:
            self.core.eval()
        return self

    def adapt(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.adapter(h, valid) if self.adapter_enabled else h

    def encode_video(
        self, h: torch.Tensor, valid: torch.Tensor
    ) -> tuple[VideoEncoding, torch.Tensor]:
        h_prime = self.adapt(h, valid)
        # Never no_grad this call: gradient must cross the frozen core into the adapter.
        return self.bridge.encode_video(h_prime, valid), h_prime

    def encode_text(
        self, ids: torch.Tensor, segments: torch.Tensor, mask: torch.Tensor
    ) -> TextEncoding:
        return self.bridge.encode_text(ids, segments, mask)

    def inference_state_dict(self) -> dict[str, Any]:
        return {
            "format": "elsc-inference-v1",
            "core": self.core.state_dict(),
            "adapter": self.adapter.state_dict() if self.adapter_enabled else None,
            "adapter_enabled": self.adapter_enabled,
        }
