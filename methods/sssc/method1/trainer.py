from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .baseline import baseline_loss_from_local, encode_local
from .config import AuxiliaryConfig
from .distributed import DistributedRuntime, ddp_weighted_auxiliary
from .losses.shared_support import reference_support, span_contrast_terms
from .sampling import cyclically_shift_support
from .schemas import SchemaError


_BASE_ONLY_ARMS = {"base_initial", "base_continuation"}
_SPAN_ARMS = {
    "span_shared",
    "span_shared_gate",
    "span_independent",
    "span_independent_gate",
    "span_random_support",
}


class Method1TrainModel(nn.Module):
    """DDP-visible owner of the one student forward and combined objective."""

    def __init__(
        self,
        student: nn.Module,
        auxiliary_config: AuxiliaryConfig,
        runtime: DistributedRuntime,
        *,
        baseline_seed: int,
    ) -> None:
        super().__init__()
        if auxiliary_config.arm not in _BASE_ONLY_ARMS | _SPAN_ARMS:
            raise ValueError(
                f"arm {auxiliary_config.arm!r} needs its explicitly implemented control scorer"
            )
        self.student = student
        self.config = auxiliary_config
        self.runtime = runtime
        self.baseline_seed = int(baseline_seed)

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
        )
        if self.config.arm in _BASE_ONLY_ARMS:
            return {"loss": base, "base": base.detach()}

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
