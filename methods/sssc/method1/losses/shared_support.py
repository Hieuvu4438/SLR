from __future__ import annotations

import torch
import torch.nn.functional as F

from ..schemas import AuxiliaryTerms, SchemaError, require_tensor


def _validate_inputs(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    q_pos: torch.Tensor,
    q_neg: torch.Tensor,
    video_valid: torch.Tensor,
    edit_valid: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    require_tensor("x", x, shape=(None, None, None), finite=True)
    batch, clips, dim = x.shape
    require_tensor("x_ref", x_ref, shape=(batch, clips, dim), finite=True)
    require_tensor("q_pos", q_pos, shape=(batch, None, None, dim), finite=True)
    _, negatives, edits, _ = q_pos.shape
    require_tensor("q_neg", q_neg, shape=(batch, negatives, edits, dim), finite=True)
    require_tensor("video_valid", video_valid, shape=(batch, clips), dtype=torch.bool)
    require_tensor("edit_valid", edit_valid, shape=(batch, negatives, edits), dtype=torch.bool)
    require_tensor("confidence", confidence, shape=(batch, negatives, edits), finite=True)
    if not bool(video_valid.any(dim=-1).all()):
        raise SchemaError("each video must contain at least one valid local clip")
    if bool((confidence < 0).any()):
        raise SchemaError("confidence weights must be non-negative")
    if bool(edit_valid.any()):
        selected_pos = q_pos[edit_valid]
        selected_neg = q_neg[edit_valid]
        if bool((selected_pos.norm(dim=-1) <= 1e-6).any()) or bool(
            (selected_neg.norm(dim=-1) <= 1e-6).any()
        ):
            raise SchemaError("valid edits must have nonzero positive/negative span vectors")
        normalized_pos = F.normalize(selected_pos.float(), dim=-1, eps=1e-6)
        normalized_neg = F.normalize(selected_neg.float(), dim=-1, eps=1e-6)
        if bool(((normalized_pos - normalized_neg).norm(dim=-1) <= 1e-6).any()):
            raise SchemaError("valid edit has a near-zero normalized text difference")
    return batch, clips, negatives, edits, dim


def reference_support(
    x_ref: torch.Tensor,
    q: torch.Tensor,
    video_valid: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("support temperature must be positive")
    batch, clips, dim = x_ref.shape
    if q.ndim != 4 or q.shape[0] != batch or q.shape[-1] != dim:
        raise SchemaError("q must be [B,K,E,D] and align with x_ref")
    if video_valid.shape != (batch, clips) or video_valid.dtype != torch.bool:
        raise SchemaError("video_valid must be bool [B,L]")
    if not bool(video_valid.any(dim=-1).all()):
        raise SchemaError("reference support requires at least one valid clip per video")
    if not bool(torch.isfinite(x_ref).all()) or not bool(torch.isfinite(q).all()):
        raise SchemaError("reference support inputs contain NaN or infinity")
    compute_dtype = torch.float64 if x_ref.dtype == q.dtype == torch.float64 else torch.float32
    with torch.no_grad():
        reference = F.normalize(x_ref.detach().to(compute_dtype), dim=-1, eps=1e-6)
        spans = F.normalize(q.detach().to(compute_dtype), dim=-1, eps=1e-6)
        logits = torch.einsum("bld,bked->bkel", reference, spans) / float(tau)
        logits = logits.masked_fill(~video_valid[:, None, None, :], float("-inf"))
        support = logits.softmax(dim=-1)
        support = support.masked_fill(~video_valid[:, None, None, :], 0.0)
    if not bool(torch.isfinite(support).all()):
        raise FloatingPointError("reference support produced NaN or infinity")
    return support


def span_contrast_terms(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    q_pos: torch.Tensor,
    q_neg: torch.Tensor,
    video_valid: torch.Tensor,
    edit_valid: torch.Tensor,
    confidence: torch.Tensor,
    *,
    mode: str,
    tau: float,
    margin: float,
    random_support: torch.Tensor | None = None,
) -> AuxiliaryTerms:
    _validate_inputs(x, x_ref, q_pos, q_neg, video_valid, edit_valid, confidence)
    if margin < 0:
        raise ValueError("margin must be non-negative")
    compute_dtype = (
        torch.float64
        if x.dtype == x_ref.dtype == q_pos.dtype == q_neg.dtype == torch.float64
        else torch.float32
    )
    student = F.normalize(x.to(compute_dtype), dim=-1, eps=1e-6)
    positive = F.normalize(q_pos.detach().to(compute_dtype), dim=-1, eps=1e-6)
    negative = F.normalize(q_neg.detach().to(compute_dtype), dim=-1, eps=1e-6)
    positive_support = reference_support(x_ref, positive, video_valid, tau)

    if mode == "shared":
        applied_support = positive_support
        delta = torch.einsum(
            "bkel,bld,bked->bke", applied_support, student, positive - negative
        )
    elif mode == "independent":
        negative_support = reference_support(x_ref, negative, video_valid, tau)
        applied_support = positive_support
        positive_score = torch.einsum(
            "bkel,bld,bked->bke", positive_support, student, positive
        )
        negative_score = torch.einsum(
            "bkel,bld,bked->bke", negative_support, student, negative
        )
        delta = positive_score - negative_score
    elif mode == "random":
        if random_support is None or random_support.shape != positive_support.shape:
            raise SchemaError("random mode requires a [B,K,E,L] random_support tensor")
        if random_support.requires_grad or not bool(torch.isfinite(random_support).all()):
            raise SchemaError("random support must be finite and detached")
        if not torch.equal(
            random_support.masked_select(~video_valid[:, None, None, :]),
            torch.zeros_like(random_support.masked_select(~video_valid[:, None, None, :])),
        ):
            raise SchemaError("random support assigns mass to invalid clips")
        applied_support = random_support.detach().float()
        delta = torch.einsum(
            "bkel,bld,bked->bke", applied_support, student, positive - negative
        )
    else:
        raise ValueError(f"unsupported support mode: {mode}")

    weights = edit_valid.float() * confidence.detach().float()
    penalty = F.relu(float(margin) - delta)
    numerator = (weights * penalty).sum() + 0.0 * x.sum()
    denominator = weights.sum().detach()
    if not bool(torch.isfinite(numerator)) or not bool(torch.isfinite(denominator)):
        raise FloatingPointError("auxiliary reduction produced NaN or infinity")
    return AuxiliaryTerms(
        numerator=numerator,
        denominator=denominator,
        diagnostics={
            "delta": delta.detach(),
            "active": ((delta < margin) & edit_valid).detach(),
            "support": applied_support.detach(),
            "positive_support": positive_support.detach(),
            "weight": weights.detach(),
        },
    )
