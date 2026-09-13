from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .baseline import baseline_loss_from_local, encode_local
from .config import AuxiliaryConfig
from .distributed import DistributedRuntime, ddp_weighted_auxiliary
from .losses.shared_support import reference_support, span_contrast_terms
from .losses.strong_controls import (
    caption_hard_negative_terms,
    fsc_local_hard_negative_terms,
)
from .sampling import cyclically_shift_support
from .schemas import SchemaError


_BASE_ONLY_ARMS = {"base_initial", "base_continuation"}
_SPAN_ARMS = {
    "span_shared",
    "span_independent",
    "span_random_support",
}
_STRONG_CONTROL_ARMS = {"caption_hn", "fsc_local", "fsc_local_caption_hn"}


class Method1TrainModel(nn.Module):
    """DDP-visible owner of the one student forward and combined objective."""

    def __init__(
        self,
        student: nn.Module,
        auxiliary_config: AuxiliaryConfig,
        runtime: DistributedRuntime,
        *,
        baseline_seed: int,
        inner_similarity_temperature: float = 0.07,
        checkpoint_score_blocks: bool = False,
        video_pair_block: int = 32,
        text_pair_block: int = 64,
    ) -> None:
        super().__init__()
        if auxiliary_config.arm not in _BASE_ONLY_ARMS | _SPAN_ARMS | _STRONG_CONTROL_ARMS:
            raise ValueError(
                f"arm {auxiliary_config.arm!r} needs its explicitly implemented control scorer"
            )
        self.student = student
        self.config = auxiliary_config
        self.runtime = runtime
        self.baseline_seed = int(baseline_seed)
        self.inner_similarity_temperature = float(inner_similarity_temperature)
        self.checkpoint_score_blocks = bool(checkpoint_score_blocks)
        self.video_pair_block = int(video_pair_block)
        self.text_pair_block = int(text_pair_block)

    def forward(
        self,
        batch: dict[str, Any],
        *,
        optimizer_step: int,
        microstep: int = 0,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        encoding = encode_local(self.student, batch)
        base = baseline_loss_from_local(
            self.student,
            encoding,
            runtime=self.runtime,
            seed=self.baseline_seed,
            optimizer_step=optimizer_step,
            microstep=microstep,
            temperature=self.inner_similarity_temperature,
            checkpoint_score_blocks=self.checkpoint_score_blocks,
            video_block=self.video_pair_block,
            text_block=self.text_pair_block,
        )
        if self.config.arm in _BASE_ONLY_ARMS:
            return {"loss": base, "base": base.detach()}

        if self.config.arm in _STRONG_CONTROL_ARMS:
            return self._strong_control_forward(batch, encoding, base)

        required = {"x_ref", "q_pos", "q_neg", "edit_valid", "confidence"}
        missing = sorted(required - set(batch))
        if missing:
            raise SchemaError(f"auxiliary batch fields are missing: {', '.join(missing)}")
        student_video = encoding.video_raw[:, 1:, :]
        video_valid = ~encoding.video_ignore_raw[:, 1:].bool()
        confidence = (
            batch["confidence"]
            if self.config.reliability_gate
            else torch.ones_like(batch["confidence"])
        )
        random_support = None
        if self.config.support_mode == "random":
            identity_fields = {"video_uid", "edit_uids"}
            identity_missing = sorted(identity_fields - set(batch))
            if identity_missing:
                raise SchemaError(
                    "random-support identities are missing: " + ", ".join(identity_missing)
                )
            positive_support = reference_support(
                batch["x_ref"], batch["q_pos"], video_valid, self.config.tau_support
            )
            random_support = cyclically_shift_support(
                positive_support,
                video_valid,
                seed=self.baseline_seed,
                epoch=epoch,
                video_uids=batch["video_uid"],
                edit_uids=batch["edit_uids"],
            )
        terms = span_contrast_terms(
            student_video,
            batch["x_ref"],
            batch["q_pos"],
            batch["q_neg"],
            video_valid,
            batch["edit_valid"],
            confidence,
            mode=self.config.support_mode,
            tau=self.config.tau_support,
            margin=self.config.margin,
            random_support=random_support,
        )
        auxiliary_backward, auxiliary_logs = ddp_weighted_auxiliary(terms, self.runtime)
        loss = base + float(self.config.aux_weight) * auxiliary_backward
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("combined Method 1 loss is nonfinite")
        return {
            "loss": loss,
            "base": base.detach(),
            **auxiliary_logs,
        }

    def _strong_control_forward(
        self,
        batch: dict[str, Any],
        encoding,
        base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        required = {"negative_input_ids", "negative_text_valid", "edit_valid"}
        missing = sorted(required - set(batch))
        if missing:
            raise SchemaError(f"strong-control batch fields are missing: {', '.join(missing)}")
        negative_ids = batch["negative_input_ids"]
        negative_valid = batch["negative_text_valid"].bool()
        if negative_ids.ndim != 3 or negative_valid.shape != negative_ids.shape:
            raise SchemaError("negative input IDs/masks must align as [B,K,M]")
        batch_size, negatives, positions = negative_ids.shape
        result = self.student.get_text_feat(
            negative_ids.reshape(batch_size * negatives, positions),
            torch.zeros_like(negative_ids).reshape(batch_size * negatives, positions),
            negative_valid.reshape(batch_size * negatives, positions),
            shaped=False,
            get_hidden=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("UPRet negative text encoding must return mask, tokens, class")
        returned_valid, negative_tokens, _ = result
        returned_valid = returned_valid.bool().view(batch_size, negatives, positions)
        if not torch.equal(returned_valid, negative_valid):
            raise SchemaError("negative text encoder mask disagrees with cached tokenization")
        negative_tokens = negative_tokens.view(
            batch_size, negatives, positions, negative_tokens.shape[-1]
        )
        caption_tokens = torch.cat((encoding.text_raw[:, None, :, :], negative_tokens), dim=1)
        caption_valid = torch.cat(
            (encoding.text_valid[:, None, :], negative_valid), dim=1
        )
        negative_class_valid = batch["edit_valid"].bool().any(dim=-1)
        video_tokens = encoding.video_raw[:, 1:, :]
        video_valid = ~encoding.video_ignore_raw[:, 1:].bool()
        output: dict[str, torch.Tensor] = {"base": base.detach()}
        output["negative_encoding_slots"] = torch.tensor(
            batch_size * negatives, device=base.device, dtype=torch.int64
        )
        output["valid_negative_count"] = negative_class_valid.sum().detach()
        total = base
        if self.config.arm in {"caption_hn", "fsc_local_caption_hn"}:
            caption_terms = caption_hard_negative_terms(
                video_tokens,
                video_valid,
                caption_tokens,
                caption_valid,
                negative_class_valid,
                logit_scale=self.student.clip.logit_scale,
                temperature=self.inner_similarity_temperature,
            )
            caption_backward, caption_logs = ddp_weighted_auxiliary(
                caption_terms, self.runtime
            )
            total = total + float(self.config.caption_loss_weight) * caption_backward
            output.update(
                {
                    "caption_auxiliary": caption_logs["auxiliary"],
                    "caption_weight_count": caption_logs["auxiliary_weight_count"],
                }
            )
        if self.config.arm in {"fsc_local", "fsc_local_caption_hn"}:
            fsc_terms = fsc_local_hard_negative_terms(
                video_tokens,
                video_valid,
                caption_tokens,
                caption_valid,
                negative_class_valid,
                logit_scale=self.student.clip.logit_scale,
                loss_name=self.config.fsc_loss_name,
                focal_gamma=self.config.fsc_focal_gamma,
                label_smoothing=self.config.fsc_label_smoothing,
            )
            fsc_backward, fsc_logs = ddp_weighted_auxiliary(fsc_terms, self.runtime)
            total = total + float(self.config.fsc_loss_weight) * fsc_backward
            output.update(
                {
                    "fsc_auxiliary": fsc_logs["auxiliary"],
                    "fsc_weight_count": fsc_logs["auxiliary_weight_count"],
                }
            )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("combined strong-control loss is nonfinite")
        output["loss"] = total
        return output


def complete_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float,
    scaler: Any | None = None,
) -> torch.Tensor:
    """Apply upstream's outer clip, BertAdam step and CLIP logit-scale clamp."""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if scaler is not None:
        scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    target = model.module if hasattr(model, "module") else model
    student = target.student if isinstance(target, Method1TrainModel) else target
    if not hasattr(student, "clip") or not hasattr(student.clip, "logit_scale"):
        raise SchemaError("student is missing clip.logit_scale")
    with torch.no_grad():
        student.clip.logit_scale.clamp_(max=float(torch.log(torch.tensor(100.0))))
    optimizer.zero_grad(set_to_none=True)
    return gradient_norm.detach()
