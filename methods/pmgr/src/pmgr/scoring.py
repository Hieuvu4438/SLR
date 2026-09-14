from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor


class ScoringContractError(ValueError):
    """Raised when an input would change the PMGR score semantics."""


@dataclass(frozen=True)
class EncodedPMGRBatch:
    video_hidden: Tensor
    video_valid: Tensor
    text_hidden: Tensor
    text_valid: Tensor
    aug_hidden: Tensor
    aug_valid: Tensor

    def validate(self) -> None:
        _validate_inputs(
            self.video_hidden,
            self.text_hidden,
            self.aug_hidden,
            self.video_valid,
            self.text_valid,
            self.aug_valid,
        )


def video_valid_from_legacy(mask: Tensor, *, hidden_length: int | None = None) -> Tensor:
    """Convert CiCo's 1=excluded video mask exactly once and exclude visual CLS."""
    if mask.ndim != 2:
        raise ScoringContractError("legacy video mask must have shape [videos, positions]")
    if mask.dtype == torch.bool:
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ScoringContractError("legacy video mask must be binary")
    elif not bool(((mask == 0) | (mask == 1)).all()):
        raise ScoringContractError("legacy video mask must be binary")
    if hidden_length is not None and mask.shape[1] != hidden_length:
        raise ScoringContractError("legacy video mask/token length mismatch")
    if mask.shape[1] < 2:
        raise ScoringContractError("video sequence must include CLS and at least one feature slot")
    valid = mask.eq(0)
    valid = valid.clone()
    valid[:, 0] = False
    if not bool(valid.any(dim=1).all()):
        raise ScoringContractError("every video needs at least one valid non-CLS position")
    return valid


def text_valid_from_cico(mask: Tensor, *, hidden_length: int | None = None) -> Tensor:
    """Normalize CiCo's EOT-derived 1=valid text mask."""
    if mask.ndim != 2 or not bool(((mask == 0) | (mask == 1)).all()):
        raise ScoringContractError("CiCo text mask must be a binary [groups, positions] tensor")
    if hidden_length is not None and mask.shape[1] != hidden_length:
        raise ScoringContractError("text mask/token length mismatch")
    valid = mask.eq(1)
    if not bool(valid.any(dim=1).all()):
        raise ScoringContractError("every text needs at least one valid position")
    return valid


def _validate_inputs(
    video: Tensor,
    text: Tensor,
    augmented_text: Tensor,
    video_valid: Tensor,
    text_valid: Tensor,
    augmented_valid: Tensor,
) -> None:
    if any(value.ndim != 3 for value in (video, text, augmented_text)):
        raise ScoringContractError("hidden tensors must have [items, positions, dimensions]")
    if video.shape[0] == 0 or text.shape[0] == 0:
        raise ScoringContractError("score matrix cannot be empty")
    if text.shape[0] != augmented_text.shape[0]:
        raise ScoringContractError("original and augmented group counts differ")
    if len({video.shape[-1], text.shape[-1], augmented_text.shape[-1]}) != 1:
        raise ScoringContractError("embedding dimensions differ")
    for hidden, valid, name in (
        (video, video_valid, "video"),
        (text, text_valid, "text"),
        (augmented_text, augmented_valid, "augmented text"),
    ):
        if valid.dtype != torch.bool or valid.shape != hidden.shape[:2]:
            raise ScoringContractError(f"{name} mask must be bool with True=valid")
        if valid.device != hidden.device:
            raise ScoringContractError(f"{name} mask and hidden tensor must share a device")
        if not bool(valid.any(dim=1).all()):
            raise ScoringContractError(f"every {name} sequence needs valid positions")
        if not bool(torch.isfinite(hidden).all()):
            raise ScoringContractError(f"{name} hidden tensor contains NaN or infinity")


def mixed_pair_scores(
    video: Tensor,
    text: Tensor,
    augmented_text: Tensor,
    video_valid: Tensor,
    text_valid: Tensor,
    augmented_valid: Tensor,
    *,
    omega: float = 0.5,
    sigma: float = 0.07,
    normalize_eps: float = 1e-6,
    mask_policy: str = "valid_tokens_only",
) -> tuple[Tensor, Tensor, Tensor]:
    """Return unscaled mixed, video-averaged, and text-averaged CiCo channels.

    Every returned matrix is oriented ``[num_videos, num_text_groups]``. The corrected
    ``valid_tokens_only`` policy excludes invalid positions from both correspondence softmaxes.
    ``legacy_unmasked`` is retained only for matched compatibility measurements.
    """
    if not 0.0 <= omega <= 1.0 or sigma <= 0 or normalize_eps <= 0:
        raise ScoringContractError("invalid score mixing, smoothing, or normalization value")
    if mask_policy not in {"valid_tokens_only", "legacy_unmasked"}:
        raise ScoringContractError(f"unsupported score mask policy: {mask_policy}")
    _validate_inputs(video, text, augmented_text, video_valid, text_valid, augmented_valid)

    work_dtype = (
        torch.float64
        if any(value.dtype == torch.float64 for value in (video, text, augmented_text))
        else torch.float32
    )
    v = F.normalize(video.to(work_dtype), dim=-1, eps=normalize_eps)
    t = F.normalize(text.to(work_dtype), dim=-1, eps=normalize_eps)
    ta = F.normalize(augmented_text.to(work_dtype), dim=-1, eps=normalize_eps)

    correspondence = torch.einsum("ild,qmd->iqlm", v, t)
    logits_a = correspondence / sigma
    if mask_policy == "valid_tokens_only":
        logits_a = logits_a.masked_fill(~text_valid[None, :, None, :], -torch.inf)
    per_video_position = (torch.softmax(logits_a, dim=-1) * correspondence).sum(dim=-1)
    per_video_position = per_video_position.masked_fill(~video_valid[:, None, :], 0.0)
    a = per_video_position.sum(dim=-1) / video_valid.sum(dim=-1)[:, None]

    augmented_correspondence = torch.einsum("ild,qmd->iqlm", v, ta)
    logits_b = augmented_correspondence / sigma
    if mask_policy == "valid_tokens_only":
        logits_b = logits_b.masked_fill(~video_valid[:, None, :, None], -torch.inf)
    per_text_position = (
        torch.softmax(logits_b, dim=-2) * augmented_correspondence
    ).sum(dim=-2)
    per_text_position = per_text_position.masked_fill(~augmented_valid[None, :, :], 0.0)
    b = per_text_position.sum(dim=-1) / augmented_valid.sum(dim=-1)[None, :]

    q = omega * a + (1.0 - omega) * b
    if not bool(torch.isfinite(q).all()):
        raise ScoringContractError("pair scorer produced a nonfinite value")
    return q, a, b


def _block_slices(length: int, size: int) -> Iterator[slice]:
    if size < 1:
        raise ScoringContractError("block size must be positive")
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def mixed_pair_scores_blocked(
    video: Tensor,
    text: Tensor,
    augmented_text: Tensor,
    video_valid: Tensor,
    text_valid: Tensor,
    augmented_valid: Tensor,
    *,
    video_block: int,
    text_block: int,
    omega: float = 0.5,
    sigma: float = 0.07,
    normalize_eps: float = 1e-6,
    mask_policy: str = "valid_tokens_only",
) -> tuple[Tensor, Tensor, Tensor]:
    rows_q: list[Tensor] = []
    rows_a: list[Tensor] = []
    rows_b: list[Tensor] = []
    for row in _block_slices(video.shape[0], video_block):
        q_columns: list[Tensor] = []
        a_columns: list[Tensor] = []
        b_columns: list[Tensor] = []
        for column in _block_slices(text.shape[0], text_block):
            q, a, b = mixed_pair_scores(
                video[row],
                text[column],
                augmented_text[column],
                video_valid[row],
                text_valid[column],
                augmented_valid[column],
                omega=omega,
                sigma=sigma,
                normalize_eps=normalize_eps,
                mask_policy=mask_policy,
            )
            q_columns.append(q)
            a_columns.append(a)
            b_columns.append(b)
        rows_q.append(torch.cat(q_columns, dim=1))
        rows_a.append(torch.cat(a_columns, dim=1))
        rows_b.append(torch.cat(b_columns, dim=1))
    return torch.cat(rows_q), torch.cat(rows_a), torch.cat(rows_b)
