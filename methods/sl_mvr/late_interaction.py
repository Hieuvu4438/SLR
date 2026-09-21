"""Direct multi-vector retrieval over contextual sign windows and text tokens.

The module is independent of SEDS.  It consumes precomputed visual token banks
and contextual text-token banks, projects them into a shared space, and returns
one video-by-text score matrix.  Invalid padded tokens never participate.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_tokens(tokens: Tensor, mask: Tensor, name: str) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, tokens, channels]")
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"{name}_mask must have shape {tuple(tokens.shape[:2])}")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name}_mask must be boolean")
    if not torch.all(mask.any(dim=1)):
        raise ValueError(f"every {name} sample needs at least one valid token")


def _project_normalize(tokens: Tensor, projection: nn.Module) -> Tensor:
    return F.normalize(projection(tokens).float(), dim=-1)


def mean_maxsim(
    video_tokens: Tensor,
    text_tokens: Tensor,
    video_mask: Tensor,
    text_mask: Tensor,
) -> Tensor:
    """Return video-by-text MeanMaxSim scores.

    Each contextual text token selects its best supported visual window.  The
    aggregation is intentionally text-to-video asymmetric: transition windows
    need not match a spoken-language token.  The same score matrix is ranked by
    columns for T2V and by rows for V2T.
    """
    _validate_tokens(video_tokens, video_mask, "video")
    _validate_tokens(text_tokens, text_mask, "text")
    if video_tokens.shape[-1] != text_tokens.shape[-1]:
        raise ValueError("video and text projection dimensions must match")

    # [text query, video candidate, text token, video token]
    similarities = torch.einsum("qld,vtd->qvlt", text_tokens, video_tokens)
    valid_video = video_mask[None, :, None, :]
    floor = torch.finfo(similarities.dtype).min
    similarities = similarities.masked_fill(~valid_video, floor)
    best_visual = similarities.max(dim=-1).values
    valid_text = text_mask[:, None, :]
    summed = (best_visual * valid_text).sum(dim=-1)
    counts = valid_text.sum(dim=-1).clamp_min(1)
    return (summed / counts).transpose(0, 1).contiguous()


def mean_pool_similarity(
    video_tokens: Tensor,
    text_tokens: Tensor,
    video_mask: Tensor,
    text_mask: Tensor,
) -> Tensor:
    """Matched single-vector control returning a video-by-text score matrix."""
    _validate_tokens(video_tokens, video_mask, "video")
    _validate_tokens(text_tokens, text_mask, "text")

    def pool(tokens: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return F.normalize(pooled.float(), dim=-1)

    return pool(video_tokens, video_mask) @ pool(text_tokens, text_mask).T


class DualLevelLateInteraction(nn.Module):
    """Two complementary visual levels trained with independent sigmoid losses."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        projection_dim: int = 256,
        *,
        init_logit_scale: float = math.log(10.0),
        init_logit_bias: float = -10.0,
    ) -> None:
        super().__init__()
        if min(visual_dim, text_dim, projection_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.visual_feature_projection = nn.Linear(visual_dim, projection_dim, bias=False)
        self.visual_latent_projection = nn.Linear(visual_dim, projection_dim, bias=False)
        self.text_feature_projection = nn.Linear(text_dim, projection_dim, bias=False)
        self.text_latent_projection = nn.Linear(text_dim, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_logit_bias)))

    def encode_video(self, visual_features: Tensor, visual_latents: Tensor) -> tuple[Tensor, Tensor]:
        if visual_features.shape != visual_latents.shape:
            raise ValueError("visual_features and visual_latents must have identical shapes")
        return (
            _project_normalize(visual_features, self.visual_feature_projection),
            _project_normalize(visual_latents, self.visual_latent_projection),
        )

    def encode_text(self, text_tokens: Tensor) -> tuple[Tensor, Tensor]:
        return (
            _project_normalize(text_tokens, self.text_feature_projection),
            _project_normalize(text_tokens, self.text_latent_projection),
        )

    def score_encoded(
        self,
        visual_features: Tensor,
        visual_latents: Tensor,
        text_features: Tensor,
        text_latents: Tensor,
        video_mask: Tensor,
        text_mask: Tensor,
        *,
        interaction: str = "late",
    ) -> dict[str, Tensor]:
        if interaction == "late":
            scorer = mean_maxsim
        elif interaction == "mean":
            scorer = mean_pool_similarity
        else:
            raise ValueError("interaction must be 'late' or 'mean'")
        feature_scores = scorer(visual_features, text_features, video_mask, text_mask)
        latent_scores = scorer(visual_latents, text_latents, video_mask, text_mask)
        return {
            "feature_scores": feature_scores,
            "latent_scores": latent_scores,
            "scores": feature_scores + latent_scores,
        }

    def forward(
        self,
        visual_features: Tensor,
        visual_latents: Tensor,
        text_tokens: Tensor,
        video_mask: Tensor,
        text_mask: Tensor,
        *,
        interaction: str = "late",
    ) -> dict[str, Tensor]:
        feature_video, latent_video = self.encode_video(visual_features, visual_latents)
        feature_text, latent_text = self.encode_text(text_tokens)
        return self.score_encoded(
            feature_video,
            latent_video,
            feature_text,
            latent_text,
            video_mask,
            text_mask,
            interaction=interaction,
        )

    def sigmoid_loss(self, scores: Tensor) -> Tensor:
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("in-batch sigmoid loss requires a square score matrix")
        targets = torch.full_like(scores, -1.0)
        targets.fill_diagonal_(1.0)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits = scale * scores + self.logit_bias
        return F.softplus(-targets * logits).mean()

    def loss(self, outputs: dict[str, Tensor]) -> Tensor:
        """Dual loss keeps both representation levels individually retrievable."""
        return 0.5 * (
            self.sigmoid_loss(outputs["feature_scores"])
            + self.sigmoid_loss(outputs["latent_scores"])
        )
