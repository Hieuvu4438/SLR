from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VideoEncoding:
    mask: torch.Tensor
    tokens: torch.Tensor
    cls: torch.Tensor


@dataclass
class TextEncoding:
    mask: torch.Tensor
    tokens: torch.Tensor
    cls: torch.Tensor


class CiCoBridge:
    """Strict tensor adapter around a pinned CiCo ``CLIP4Clip`` instance."""

    def __init__(self, core: Any):
        self.core = core

    @staticmethod
    def upstream_video(h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError("h must have shape [B,F,D]")
        return h.transpose(1, 2).unsqueeze(-1).contiguous()

    @staticmethod
    def upstream_video_mask(valid: torch.Tensor) -> torch.Tensor:
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("valid must be a bool tensor with shape [B,F]")
        mask = torch.ones(valid.shape[0], valid.shape[1] + 1, dtype=torch.long, device=valid.device)
        mask[:, 1:] = (~valid).long()
        return mask

    def encode_video(self, h: torch.Tensor, valid: torch.Tensor) -> VideoEncoding:
        mask = self.upstream_video_mask(valid)
        result = self.core.get_visual_output(
            self.upstream_video(h), mask, shaped=True, video_frame=1, get_hidden=True
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("unexpected CiCo get_visual_output return contract")
        returned_mask, tokens, cls = result
        return VideoEncoding(returned_mask, tokens, cls)

    def encode_text(
        self, ids: torch.Tensor, segments: torch.Tensor, input_mask: torch.Tensor
    ) -> TextEncoding:
        result = self.core.get_sequence_output(
            ids, segments, input_mask, shaped=False, get_hidden=True
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("unexpected CiCo get_sequence_output return contract")
        returned_mask, tokens, cls = result
        return TextEncoding(returned_mask, tokens, cls)

    def score(
        self,
        video: VideoEncoding,
        text: TextEncoding,
        *,
        text_aug: TextEncoding | None = None,
        objective: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict[str, Any] = {}
        if text_aug is not None:
            kwargs.update(sequence_hidden_aug=text_aug.tokens, text_mask_aug=text_aug.mask)
        result = self.core.get_similarity_logits(
            text.tokens,
            video.tokens,
            text.mask,
            video.mask,
            shaped=True,
            loose_type=getattr(self.core, "loose_type", True),
            is_train=objective,
            **kwargs,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("CiCo scorer did not return I2T and T2I matrices")
        i2t, t2i = result[:2]
        if i2t.ndim != 2 or t2i.shape != i2t.shape:
            raise RuntimeError(f"CiCo score orientation violation: {i2t.shape}, {t2i.shape}")
        return i2t, t2i

    @staticmethod
    def mixed_score(i2t: torch.Tensor, t2i: torch.Tensor, dual_mix: float) -> torch.Tensor:
        if i2t.shape != t2i.shape:
            raise ValueError("I2T and T2I must both be [N_video,N_text]")
        return dual_mix * i2t + (1.0 - dual_mix) * t2i

    def paired_score(
        self,
        video: VideoEncoding,
        text: TextEncoding,
        *,
        dual_mix: float,
    ) -> torch.Tensor:
        if video.tokens.shape[0] != text.tokens.shape[0]:
            raise ValueError("paired_score requires aligned video/text batches")
        if getattr(self.core, "sim_header", None) == "Filip":
            # This is the diagonal of CiCo ``flip_similarity_softmax`` without
            # materializing its [B,B,F,T] all-pairs tensor.  Auxiliary evidence
            # pairs are local to a rank, so no distributed gather belongs here.
            video_tokens = video.tokens / video.tokens.norm(dim=-1, keepdim=True)
            text_tokens = text.tokens / text.tokens.norm(dim=-1, keepdim=True)
            aligned = torch.einsum("bfs,bts->bft", video_tokens, text_tokens)

            video_valid = video.mask == 0
            text_valid = text.mask == 1
            if video_valid.shape != aligned.shape[:2]:
                raise ValueError("video mask/token shape mismatch in paired Filip score")
            if text_valid.shape != (aligned.shape[0], aligned.shape[2]):
                raise ValueError("text mask/token shape mismatch in paired Filip score")
            if not bool(video_valid.any(dim=1).all()) or not bool(text_valid.any(dim=1).all()):
                raise ValueError("paired Filip score requires a valid video and text token")

            i2t_token = torch.nansum(
                aligned * torch.softmax(aligned / 0.07, dim=2), dim=2
            )
            i2t_token = i2t_token.masked_fill(~video_valid, 0.0)
            i2t = i2t_token.sum(dim=1) / video_valid.sum(dim=1)

            t2i_token = torch.nansum(
                aligned * torch.softmax(aligned / 0.07, dim=1), dim=1
            )
            t2i_token = t2i_token.masked_fill(~text_valid, 0.0)
            t2i = t2i_token.sum(dim=1) / text_valid.sum(dim=1)

            scale = self.core.clip.logit_scale.exp()
            return scale * (dual_mix * i2t + (1.0 - dual_mix) * t2i)

        i2t, t2i = self.score(video, text, objective=True)
        if i2t.shape[0] != i2t.shape[1]:
            raise ValueError("paired_score requires aligned square video/text batches")
        return self.mixed_score(i2t, t2i, dual_mix).diagonal()
