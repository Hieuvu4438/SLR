"""Contract-preserving adapter for the pinned CiCo ``Filip`` execution path.

The upstream helper returns scores after multiplication by
``exp(clip.logit_scale)``.  OCEM combines its correction with the unscaled
score, so this adapter implements the same aggregation directly and exposes
the external scale separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


class CiCoAdapterError(ValueError):
    """Raised when a batch or upstream model violates the locked contract."""


@dataclass(frozen=True)
class CiCoEncoded:
    """Contextual tokens and the exact masks consumed by CiCo scoring."""

    visual_tokens: torch.Tensor
    canonical_text_tokens: torch.Tensor
    augmented_text_tokens: torch.Tensor
    upstream_video_mask: torch.Tensor
    canonical_text_mask: torch.Tensor
    augmented_text_mask: torch.Tensor
    visual_cls: torch.Tensor | None = None
    canonical_text_cls: torch.Tensor | None = None
    augmented_text_cls: torch.Tensor | None = None

    @property
    def Z(self) -> torch.Tensor:
        """Contextual local video tokens, excluding the visual CLS token."""

        return self.visual_tokens[:, 1:, :]

    @property
    def Y(self) -> torch.Tensor:
        """Canonical contextual text tokens (padding retained and masked)."""

        return self.canonical_text_tokens

    @property
    def video_valid(self) -> torch.Tensor:
        """Internal ``True=valid`` mask, excluding the visual CLS token."""

        return self.upstream_video_mask[:, 1:] == 0

    @property
    def text_valid(self) -> torch.Tensor:
        """Internal ``True=valid`` canonical text mask."""

        return self.canonical_text_mask == 1


@dataclass(frozen=True)
class CiCoDirectionalMatrices:
    """Raw score matrices with the common ``[video, text]`` orientation."""

    raw_t2v: torch.Tensor
    raw_v2t: torch.Tensor


@dataclass(frozen=True)
class CiCoDirectionalLogits:
    """Externally scaled directional logits with ``[video, text]`` orientation."""

    logits_t2v: torch.Tensor
    logits_v2t: torch.Tensor


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise CiCoAdapterError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


def _cross_entropy_diagonal(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise CiCoAdapterError(
            "CiCo diagonal contrastive loss requires a square score matrix"
        )
    return -torch.diagonal(torch.log_softmax(scores, dim=-1)).mean()


class CiCoAdapter:
    """Expose the actual CiCo Filip score/loss conventions without patching upstream."""

    def __init__(
        self,
        model: Any,
        *,
        dual_mix: float | None = None,
        mix_design: str | None = None,
        internal_temperature: float = 0.07,
    ) -> None:
        if not hasattr(model, "clip") or not hasattr(model.clip, "logit_scale"):
            raise CiCoAdapterError("model must expose clip.logit_scale")
        self.model = model
        self.dual_mix = float(
            getattr(model, "dual_mix", 0.5) if dual_mix is None else dual_mix
        )
        self.mix_design = str(
            getattr(model, "mix_design", "balance") if mix_design is None else mix_design
        )
        self.internal_temperature = float(internal_temperature)
        if not 0.0 <= self.dual_mix <= 1.0:
            raise CiCoAdapterError("dual_mix must be in [0, 1]")
        if self.mix_design not in {"balance", "depart"}:
            raise CiCoAdapterError("mix_design must be 'balance' or 'depart'")
        if self.internal_temperature <= 0:
            raise CiCoAdapterError("internal_temperature must be positive")

    @staticmethod
    def upstream_video_mask(video_valid: torch.Tensor) -> torch.Tensor:
        """Map ``True=valid`` to CiCo's ``CLS=1, valid=0, padding=1`` polarity."""

        _require_rank("video_valid", video_valid, 2)
        if video_valid.dtype is not torch.bool:
            raise CiCoAdapterError("video_valid must have bool dtype")
        cls = torch.ones(
            (video_valid.shape[0], 1), dtype=torch.long, device=video_valid.device
        )
        local = (~video_valid).to(dtype=torch.long)
        return torch.cat((cls, local), dim=1)

    @staticmethod
    def upstream_video_tensor(local_h: torch.Tensor) -> torch.Tensor:
        """Map ``[B,M,1024]`` features to upstream ``[B,1024,M,1]``."""

        _require_rank("local_h", local_h, 3)
        if local_h.shape[-1] != 1024:
            raise CiCoAdapterError("local_h last dimension must be 1024")
        return local_h.transpose(1, 2).unsqueeze(-1).contiguous()

    def encode(self, batch: Mapping[str, Any], *, training: bool) -> CiCoEncoded:
        """Encode an OCEM batch through the pinned CiCo text/visual entrypoints."""

        required = {"local_h", "video_valid", "text_ids", "text_input_valid"}
        missing = sorted(required.difference(batch))
        if missing:
            raise CiCoAdapterError(f"batch is missing required fields: {missing}")
        local_h = batch["local_h"]
        video_valid = batch["video_valid"]
        text_ids = batch["text_ids"]
        text_input_valid = batch["text_input_valid"]
        for name, tensor, rank in (
            ("local_h", local_h, 3),
            ("video_valid", video_valid, 2),
            ("text_ids", text_ids, 2),
            ("text_input_valid", text_input_valid, 2),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise CiCoAdapterError(f"{name} must be a torch.Tensor")
            _require_rank(name, tensor, rank)
        if video_valid.dtype is not torch.bool or text_input_valid.dtype is not torch.bool:
            raise CiCoAdapterError("internal validity masks must have bool dtype")
        if text_ids.dtype != torch.long:
            raise CiCoAdapterError("text_ids must have int64 dtype")
        if local_h.shape[:2] != video_valid.shape:
            raise CiCoAdapterError("local_h and video_valid shapes do not align")
        if text_ids.shape != text_input_valid.shape:
            raise CiCoAdapterError("text_ids and text_input_valid shapes do not align")
        if local_h.shape[0] != text_ids.shape[0]:
            raise CiCoAdapterError("video and text batch sizes differ")
        expected_feature_len = getattr(getattr(self.model.clip, "visual", None), "feature_len", None)
        if expected_feature_len is not None and local_h.shape[1] != expected_feature_len:
            raise CiCoAdapterError(
                f"CiCo checkpoint expects {expected_feature_len} local features, "
                f"got {local_h.shape[1]}"
            )

        augmented_ids = batch.get("text_aug_ids", text_ids)
        augmented_valid = batch.get("text_aug_input_valid", text_input_valid)
        if not isinstance(augmented_ids, torch.Tensor) or not isinstance(
            augmented_valid, torch.Tensor
        ):
            raise CiCoAdapterError("augmented text fields must be torch.Tensor values")
        if augmented_ids.shape != text_ids.shape or augmented_valid.shape != text_ids.shape:
            raise CiCoAdapterError("augmented and canonical text shapes differ")
        if augmented_ids.dtype != torch.long or augmented_valid.dtype is not torch.bool:
            raise CiCoAdapterError("augmented IDs/mask must be int64/bool")

        self.model.train(training)
        token_type_ids = torch.zeros_like(text_ids)
        canonical_mask, canonical_tokens, canonical_cls = self.model.get_sequence_output(
            text_ids,
            token_type_ids,
            text_input_valid.to(dtype=torch.long),
            shaped=True,
        )
        augmented_mask, augmented_tokens, augmented_cls = self.model.get_sequence_output(
            augmented_ids,
            token_type_ids,
            augmented_valid.to(dtype=torch.long),
            shaped=True,
        )
        upstream_video_mask = self.upstream_video_mask(video_valid)
        returned_video_mask, visual_tokens, visual_cls = self.model.get_visual_output(
            self.upstream_video_tensor(local_h),
            upstream_video_mask,
            shaped=True,
            video_frame=1,
        )
        if not torch.equal(returned_video_mask, upstream_video_mask):
            raise CiCoAdapterError("upstream visual encoder changed the video mask")
        return CiCoEncoded(
            visual_tokens=visual_tokens,
            canonical_text_tokens=canonical_tokens,
            augmented_text_tokens=augmented_tokens,
            upstream_video_mask=returned_video_mask,
            canonical_text_mask=canonical_mask,
            augmented_text_mask=augmented_mask,
            visual_cls=visual_cls,
            canonical_text_cls=canonical_cls,
            augmented_text_cls=augmented_cls,
        )

    def logit_scale(self) -> torch.Tensor:
        """Return CiCo's positive external score multiplier."""

        return self.model.clip.logit_scale.exp()

    def directional_matrices(self, encoded: CiCoEncoded) -> CiCoDirectionalMatrices:
        """Compute both upstream aggregations before external logit scaling."""

        visual = encoded.visual_tokens
        canonical = encoded.canonical_text_tokens
        augmented = encoded.augmented_text_tokens
        _require_rank("visual_tokens", visual, 3)
        _require_rank("canonical_text_tokens", canonical, 3)
        _require_rank("augmented_text_tokens", augmented, 3)
        if visual.shape[-1] != canonical.shape[-1] or canonical.shape != augmented.shape:
            raise CiCoAdapterError("contextual token dimensions do not align")

        # Deliberately mirror upstream division instead of F.normalize(eps=...).
        visual = visual / visual.norm(dim=-1, keepdim=True)
        canonical = canonical / canonical.norm(dim=-1, keepdim=True)
        augmented = augmented / augmented.norm(dim=-1, keepdim=True)
        affinity_t2v = torch.einsum("vmd,tld->vtml", visual, canonical)
        affinity_v2t = torch.einsum("vmd,tld->vtml", visual, augmented)

        per_video_token = torch.nansum(
            affinity_t2v
            * torch.softmax(affinity_t2v / self.internal_temperature, dim=3),
            dim=3,
        )
        video_valid = (encoded.upstream_video_mask == 0).unsqueeze(1)
        per_video_token = per_video_token.masked_fill(~video_valid, 0)
        video_count = video_valid.sum(dim=-1)
        if torch.any(video_count == 0):
            raise CiCoAdapterError("each video must contain at least one valid local token")
        raw_t2v = torch.nansum(per_video_token, dim=-1) / video_count

        per_text_token = torch.nansum(
            affinity_v2t
            * torch.softmax(affinity_v2t / self.internal_temperature, dim=2),
            dim=2,
        )
        text_valid = (encoded.augmented_text_mask == 1).unsqueeze(0)
        per_text_token = per_text_token.masked_fill(~text_valid, 0)
        text_count = text_valid.sum(dim=-1)
        if torch.any(text_count == 0):
            raise CiCoAdapterError("each augmented text must contain at least one valid token")
        raw_v2t = torch.nansum(per_text_token * text_valid, dim=-1) / text_count
        return CiCoDirectionalMatrices(raw_t2v=raw_t2v, raw_v2t=raw_v2t)

    def directional_scores(
        self, encoded: CiCoEncoded, pairs: Sequence[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(raw_t2v, raw_v2t)`` for explicit ``(video,text)`` pairs."""

        matrices = self.directional_matrices(encoded)
        if not pairs:
            empty = matrices.raw_t2v.new_empty((0,))
            return empty, empty.clone()
        video_indices = torch.tensor(
            [pair[0] for pair in pairs], device=matrices.raw_t2v.device, dtype=torch.long
        )
        text_indices = torch.tensor(
            [pair[1] for pair in pairs], device=matrices.raw_t2v.device, dtype=torch.long
        )
        return (
            matrices.raw_t2v[video_indices, text_indices],
            matrices.raw_v2t[video_indices, text_indices],
        )

    def scaled_directional_matrices(
        self, encoded: CiCoEncoded
    ) -> CiCoDirectionalLogits:
        """Recover the exact upstream scaled matrices used by its loss helper."""

        raw = self.directional_matrices(encoded)
        scale = self.logit_scale()
        return CiCoDirectionalLogits(
            logits_t2v=scale * raw.raw_t2v,
            logits_v2t=scale * raw.raw_v2t,
        )

    def legacy_eval_matrices(self, encoded: CiCoEncoded) -> CiCoDirectionalLogits:
        """Mirror released evaluation, which feeds one mixed matrix to both metrics."""

        raw = self.directional_matrices(encoded)
        mixed = self.dual_mix * raw.raw_t2v + (1.0 - self.dual_mix) * raw.raw_v2t
        scaled = self.logit_scale() * mixed
        return CiCoDirectionalLogits(logits_t2v=scaled, logits_v2t=scaled)

    def original_training_loss(
        self, encoded: CiCoEncoded, protocol: Mapping[str, Any] | None = None
    ) -> torch.Tensor:
        """Reproduce the executed upstream Filip training loss exactly."""

        dual_mix = self.dual_mix
        mix_design = self.mix_design
        if protocol is not None:
            dual_mix = float(protocol.get("dual_mix", dual_mix))
            mix_design = str(protocol.get("mix_design", mix_design))
        if not 0.0 <= dual_mix <= 1.0:
            raise CiCoAdapterError("protocol dual_mix must be in [0, 1]")
        if mix_design not in {"balance", "depart"}:
            raise CiCoAdapterError("protocol mix_design must be 'balance' or 'depart'")

        scaled = self.scaled_directional_matrices(encoded)
        i2t = scaled.logits_t2v
        t2i = scaled.logits_v2t
        if dual_mix == 1.0:
            return (_cross_entropy_diagonal(i2t) + _cross_entropy_diagonal(t2i.T)) / 2
        if mix_design == "balance":
            loss1 = dual_mix * _cross_entropy_diagonal(i2t) + (
                1.0 - dual_mix
            ) * _cross_entropy_diagonal(i2t.T)
            loss2 = dual_mix * _cross_entropy_diagonal(t2i.T) + (
                1.0 - dual_mix
            ) * _cross_entropy_diagonal(t2i)
        else:
            loss1 = dual_mix * (
                _cross_entropy_diagonal(i2t) + _cross_entropy_diagonal(i2t.T)
            )
            loss2 = (1.0 - dual_mix) * (
                _cross_entropy_diagonal(t2i.T) + _cross_entropy_diagonal(t2i)
            )
        return (loss1 + loss2) / 2
