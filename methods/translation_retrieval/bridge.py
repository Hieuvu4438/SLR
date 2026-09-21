"""Minimal pose/text bridge for a frozen Uni-Sign pose-only checkpoint.

The upstream model is supplied by the caller from its separately retained
checkout. No donor source or weights are redistributed here. This module only
exposes features before the translation decoder and adds a contrastive head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PARTS = ("body", "left", "right", "face_all")


def masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("tokens must be [B,T,D] and mask [B,T]")
    valid = mask.to(torch.bool)
    if not torch.all(valid.any(dim=1)):
        raise ValueError("every sequence needs at least one valid token")
    weights = valid.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1)


@torch.no_grad()
def pose_tokens(model: nn.Module, parts: Mapping[str, Tensor]) -> Tensor:
    """Expose Uni-Sign's pose projection without invoking its text decoder.

    The contract is four [B,T,V,3] tensors produced by Uni-Sign's own
    `load_part_kp`/collate path. This pose-only bridge intentionally rejects
    RGB-support checkpoints; their visual forward path is different.
    """
    if getattr(model.args, "rgb_support", False):
        raise ValueError("pose-only checkpoint required")
    if tuple(model.modes) != PARTS:
        raise ValueError("unexpected Uni-Sign part order")
    if any(part not in parts for part in PARTS):
        raise ValueError("missing anatomical part")

    outputs: list[Tensor] = []
    body: Tensor | None = None
    batch_time: tuple[int, int] | None = None
    for part in PARTS:
        pose = parts[part]
        if pose.ndim != 4 or pose.shape[-1] != 3:
            raise ValueError(f"{part} must have shape [B,T,V,3]")
        if batch_time is None:
            batch_time = tuple(pose.shape[:2])
        elif tuple(pose.shape[:2]) != batch_time:
            raise ValueError("anatomical parts have different batch/time shape")
        projected = model.proj_linear[part](pose).permute(0, 3, 1, 2)
        feature = model.gcn_modules[part](projected)
        if part == "body":
            body = feature
        elif part == "left":
            feature = feature + body[..., -2][..., None]
        elif part == "right":
            feature = feature + body[..., -1][..., None]
        else:
            feature = feature + body[..., 0][..., None]
        feature = model.fusion_gcn_modules[part](feature)
        outputs.append(feature.mean(dim=-1).transpose(1, 2))

    visual = torch.cat(outputs, dim=-1) + model.part_para
    return model.pose_proj(visual)


class FrozenUniSignDualEncoder(nn.Module):
    """Frozen translation encoder plus trainable global retrieval projections."""

    def __init__(self, donor: nn.Module, projection_dim: int = 256) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.donor = donor
        self.donor.requires_grad_(False)
        self.donor.eval()
        hidden_dim = donor.pose_proj.out_features
        text_dim = donor.mt5_model.config.d_model
        self.video_proj = nn.Linear(hidden_dim, projection_dim)
        self.text_proj = nn.Linear(text_dim, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592600369))  # log(1/0.07)

    def train(self, mode: bool = True) -> "FrozenUniSignDualEncoder":
        super().train(mode)
        self.donor.eval()  # keep pretrained BN/dropout deterministic
        return self

    def video_features(self, parts: Mapping[str, Tensor], mask: Tensor) -> Tensor:
        """Frozen pooled mT5-encoder video features; safe to cache by sample ID."""
        with torch.no_grad():
            visual = pose_tokens(self.donor, parts)
            if mask.shape != visual.shape[:2]:
                raise ValueError("pose attention mask does not match visual tokens")
            prompt = [f"Translate sign language video to {self.donor.lang}: "] * visual.shape[0]
            prefix = self.donor.mt5_tokenizer(
                prompt, padding=True, truncation=True, return_tensors="pt"
            ).to(visual.device)
            prefix_embeds = self.donor.mt5_model.encoder.embed_tokens(prefix["input_ids"])
            encoder_mask = torch.cat([prefix["attention_mask"], mask], dim=1)
            encoded = self.donor.mt5_model.encoder(
                inputs_embeds=torch.cat([prefix_embeds, visual], dim=1),
                attention_mask=encoder_mask,
                return_dict=True,
            ).last_hidden_state
            pooled = masked_mean(encoded[:, prefix_embeds.shape[1] :], mask)
        return pooled.float()

    def text_features(self, sentences: Sequence[str]) -> Tensor:
        """Frozen pooled mT5-encoder text features; safe to cache by caption."""
        if not sentences:
            raise ValueError("empty text batch")
        with torch.no_grad():
            batch = self.donor.mt5_tokenizer(
                list(sentences), padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            ).to(next(self.donor.parameters()).device)
            encoded = self.donor.mt5_model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            ).last_hidden_state
            pooled = masked_mean(encoded, batch["attention_mask"])
        return pooled.float()

    def encode_video(self, parts: Mapping[str, Tensor], mask: Tensor) -> Tensor:
        return F.normalize(self.video_proj(self.video_features(parts, mask)), dim=-1)

    def encode_text(self, sentences: Sequence[str]) -> Tensor:
        return F.normalize(self.text_proj(self.text_features(sentences)), dim=-1)

    def forward(
        self, parts: Mapping[str, Tensor], mask: Tensor, sentences: Sequence[str]
    ) -> tuple[Tensor, Tensor]:
        video = self.encode_video(parts, mask)
        text = self.encode_text(sentences)
        if video.shape[0] != text.shape[0]:
            raise ValueError("video/text batch mismatch")
        return video, text

    def contrastive_loss(self, video: Tensor, text: Tensor) -> Tensor:
        if video.ndim != 2 or video.shape != text.shape or video.shape[0] < 2:
            raise ValueError("paired embeddings must be [B,D] with B>=2")
        scale = self.logit_scale.exp().clamp(max=100.0)
        scores = scale * (video @ text.T)
        targets = torch.arange(scores.shape[0], device=scores.device)
        return 0.5 * (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets))
