from __future__ import annotations

import torch
import torch.nn.functional as F

from ..schemas import AuxiliaryTerms, SchemaError, require_tensor


def _validate_candidate_inputs(
    video_tokens: torch.Tensor,
    video_valid: torch.Tensor,
    caption_tokens: torch.Tensor,
    caption_valid: torch.Tensor,
    negative_valid: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    require_tensor("video_tokens", video_tokens, shape=(None, None, None), finite=True)
    batch, clips, dimension = video_tokens.shape
    require_tensor("video_valid", video_valid, shape=(batch, clips), dtype=torch.bool)
    require_tensor(
        "caption_tokens", caption_tokens, shape=(batch, None, None, dimension), finite=True
    )
    _, classes, positions, _ = caption_tokens.shape
    require_tensor(
        "caption_valid", caption_valid, shape=(batch, classes, positions), dtype=torch.bool
    )
    require_tensor("negative_valid", negative_valid, shape=(batch, classes - 1), dtype=torch.bool)
    if not bool(video_valid.any(dim=-1).all()):
        raise SchemaError("each strong-control video needs a valid local clip")
    if not bool(caption_valid.any(dim=-1).all()):
        raise SchemaError("each encoded candidate needs a finite placeholder token sequence")
    return batch, classes, clips, positions, dimension


def _candidate_loss_terms(
    logits: torch.Tensor,
    negative_valid: torch.Tensor,
    *,
    loss_name: str,
    focal_gamma: float,
    label_smoothing: float,
) -> AuxiliaryTerms:
    batch, classes = logits.shape
    class_valid = torch.cat(
        (
            torch.ones(batch, 1, dtype=torch.bool, device=logits.device),
            negative_valid.bool(),
        ),
        dim=1,
    )
    eligible_rows = negative_valid.any(dim=1)
    masked_logits = logits.float().masked_fill(~class_valid, float("-inf"))
    log_probabilities = F.log_softmax(masked_logits, dim=1)
    probabilities = log_probabilities.exp()
    class_count = class_valid.sum(dim=1, keepdim=True).float()
    targets = label_smoothing * class_valid.float() / class_count
    targets[:, 0] += 1.0 - label_smoothing
    if loss_name == "cross_entropy":
        per_class = -targets * torch.where(
            class_valid, log_probabilities, torch.zeros_like(log_probabilities)
        )
    elif loss_name == "focal_loss":
        focal = (1.0 - probabilities).pow(float(focal_gamma))
        per_class = -targets * focal * torch.where(
            class_valid, log_probabilities, torch.zeros_like(log_probabilities)
        )
    else:
        raise ValueError(f"unsupported candidate loss: {loss_name}")
    per_row = per_class.sum(dim=1)
    numerator = (per_row * eligible_rows.float()).sum() + 0.0 * logits.sum()
    denominator = eligible_rows.float().sum().detach()
    if not bool(torch.isfinite(numerator)) or not bool(torch.isfinite(denominator)):
        raise FloatingPointError("strong-control classification loss is nonfinite")
    return AuxiliaryTerms(
        numerator=numerator,
        denominator=denominator,
        diagnostics={
            "candidate_logits": masked_logits.detach(),
            "eligible_rows": eligible_rows.detach(),
            "class_valid": class_valid.detach(),
        },
    )


def caption_hard_negative_terms(
    video_tokens: torch.Tensor,
    video_valid: torch.Tensor,
    caption_tokens: torch.Tensor,
    caption_valid: torch.Tensor,
    negative_valid: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    temperature: float = 0.07,
) -> AuxiliaryTerms:
    """SAN-inspired within-example original-versus-altered caption classification."""
    _validate_candidate_inputs(
        video_tokens, video_valid, caption_tokens, caption_valid, negative_valid
    )
    if temperature <= 0:
        raise ValueError("caption matching temperature must be positive")
    video = F.normalize(video_tokens.float(), dim=-1, eps=1e-6)
    captions = F.normalize(caption_tokens.float(), dim=-1, eps=1e-6)
    affinity = torch.einsum("bld,bcmd->bclm", video, captions)
    token_valid = caption_valid[:, :, None, :].expand_as(affinity)
    probabilities = (
        (affinity / float(temperature))
        .masked_fill(~token_valid, float("-inf"))
        .softmax(dim=-1)
        .masked_fill(~token_valid, 0.0)
    )
    per_clip = (affinity * probabilities).sum(dim=-1)
    clip_weights = video_valid[:, None, :].float()
    scores = (per_clip * clip_weights).sum(dim=-1) / clip_weights.sum(dim=-1)
    logits = logit_scale.exp().float() * scores
    return _candidate_loss_terms(
        logits,
        negative_valid,
        loss_name="cross_entropy",
        focal_gamma=0.0,
        label_smoothing=0.0,
    )


def _minmax_clip_support(
    similarity: torch.Tensor, video_valid: torch.Tensor
) -> torch.Tensor:
    valid = video_valid[:, None, None, :].expand_as(similarity)
    maximum = similarity.masked_fill(~valid, float("-inf")).max(dim=-1, keepdim=True).values
    minimum = similarity.masked_fill(~valid, float("inf")).min(dim=-1, keepdim=True).values
    span = maximum - minimum
    scaled = torch.where(valid, (similarity - minimum) / span.clamp_min(1e-12), 0.0)
    total = scaled.sum(dim=-1, keepdim=True)
    uniform = valid.float() / valid.float().sum(dim=-1, keepdim=True)
    fallback = (span <= 1e-12) | (total <= 1e-12)
    support = torch.where(fallback, uniform, scaled / total.clamp_min(1e-12))
    if not bool(torch.isfinite(support).all()):
        raise FloatingPointError("FSC minmax support is nonfinite")
    return support.detach()


def fsc_local_hard_negative_terms(
    video_tokens: torch.Tensor,
    video_valid: torch.Tensor,
    caption_tokens: torch.Tensor,
    caption_valid: torch.Tensor,
    negative_valid: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    loss_name: str = "focal_loss",
    focal_gamma: float = 1.0,
    label_smoothing: float = 0.1,
) -> AuxiliaryTerms:
    """FSC-style candidate-dependent local hard-negative control with mask repairs."""
    _validate_candidate_inputs(
        video_tokens, video_valid, caption_tokens, caption_valid, negative_valid
    )
    video = F.normalize(video_tokens.float(), dim=-1, eps=1e-6)
    text_raw = caption_tokens.float()
    similarity = torch.einsum("bcmd,bld->bcml", text_raw, video)
    support = _minmax_clip_support(similarity.detach(), video_valid)
    pooled_video = F.normalize(
        torch.einsum("bcml,bld->bcmd", support, video), dim=-1, eps=1e-6
    )
    normalized_text = F.normalize(text_raw, dim=-1, eps=1e-6)
    per_token = logit_scale.exp().float() * torch.sum(pooled_video * normalized_text, dim=-1)
    scores = torch.logsumexp(
        per_token.masked_fill(~caption_valid, float("-inf")), dim=-1
    )
    terms = _candidate_loss_terms(
        scores,
        negative_valid,
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    terms.diagnostics["candidate_support"] = support.detach()
    return terms
