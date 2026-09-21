"""Frozen-feature extraction and lightweight contrastive retrieval head.

Caching is valid only while the donor pose and mT5 encoders remain frozen. The
cache deliberately stores pre-projection features so the same embeddings can
serve zero-shot and matched projection-training comparisons.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bridge import FrozenUniSignDualEncoder


@dataclass(frozen=True)
class FrozenFeatures:
    names: list[str]
    video: Tensor
    text: Tensor
    captions: list[str]

    def validate(self) -> None:
        n = len(self.names)
        if n == 0 or len(set(self.names)) != n or len(self.captions) != n:
            raise ValueError("empty, duplicate or unpaired feature IDs")
        if self.video.ndim != 2 or self.text.ndim != 2:
            raise ValueError("features must be [N,D]")
        if self.video.shape[0] != n or self.text.shape[0] != n:
            raise ValueError("feature and metadata lengths differ")
        if not torch.isfinite(self.video).all() or not torch.isfinite(self.text).all():
            raise ValueError("non-finite frozen features")


@torch.inference_mode()
def extract_frozen_features(
    bridge: FrozenUniSignDualEncoder,
    batches: Iterable[tuple[list[str], dict[str, Tensor], Tensor, list[str]]],
    device: torch.device,
) -> FrozenFeatures:
    if next(bridge.donor.parameters()).device != device:
        raise ValueError("donor must already be on the requested device")
    bridge.eval()
    names: list[str] = []
    captions: list[str] = []
    visual: list[Tensor] = []
    textual: list[Tensor] = []
    for batch_names, parts, mask, batch_captions in batches:
        if len(batch_names) != len(batch_captions):
            raise ValueError("unpaired batch")
        move_parts = {name: tensor.to(device, non_blocking=True) for name, tensor in parts.items()}
        visual.append(bridge.video_features(move_parts, mask.to(device)).cpu())
        textual.append(bridge.text_features(batch_captions).cpu())
        names.extend(batch_names)
        captions.extend(batch_captions)
    if not visual:
        raise ValueError("no batches")
    result = FrozenFeatures(names, torch.cat(visual), torch.cat(textual), captions)
    result.validate()
    return result


class ProjectionRetrievalModel(nn.Module):
    """Symmetric InfoNCE on cached frozen video/text features."""

    def __init__(self, video_dim: int, text_dim: int, output_dim: int = 256) -> None:
        super().__init__()
        if min(video_dim, text_dim, output_dim) <= 0:
            raise ValueError("projection dimensions must be positive")
        self.video_proj = nn.Linear(video_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592600369))

    def embeddings(self, video: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        if video.ndim != 2 or text.ndim != 2 or video.shape[0] != text.shape[0]:
            raise ValueError("video/text features must be paired matrices")
        return (
            F.normalize(self.video_proj(video), dim=-1),
            F.normalize(self.text_proj(text), dim=-1),
        )

    def score_matrix(self, video: Tensor, text: Tensor) -> Tensor:
        v, t = self.embeddings(video, text)
        return v @ t.T

    def score_gallery(self, video: Tensor, text: Tensor) -> Tensor:
        """Score a rectangular retrieval gallery after paired training."""
        if video.ndim != 2 or text.ndim != 2:
            raise ValueError("gallery features must be matrices")
        if video.shape[1] != self.video_proj.in_features or text.shape[1] != self.text_proj.in_features:
            raise ValueError("gallery feature widths do not match projections")
        v = F.normalize(self.video_proj(video), dim=-1)
        t = F.normalize(self.text_proj(text), dim=-1)
        return v @ t.T

    def loss(self, video: Tensor, text: Tensor) -> Tensor:
        if video.shape[0] < 2:
            raise ValueError("InfoNCE requires at least two pairs")
        v, t = self.embeddings(video, text)
        logits = self.logit_scale.exp().clamp(max=100.0) * (v @ t.T)
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


@torch.inference_mode()
def zero_shot_score_matrix(features: FrozenFeatures) -> Tensor:
    features.validate()
    if features.video.shape[1] != features.text.shape[1]:
        raise ValueError("zero-shot cosine requires a shared feature width")
    return F.normalize(features.video, dim=-1) @ F.normalize(features.text, dim=-1).T
