from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor


def _require_bool_mask(mask: Tensor, name: str, shape: tuple[int, ...]) -> None:
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must be bool with True=valid")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} shape {tuple(mask.shape)} != {shape}")


def _validate_valid_vectors(values: Tensor, mask: Tensor, name: str, eps: float) -> None:
    valid = values[mask]
    if valid.numel() == 0:
        return
    if not torch.isfinite(valid).all():
        raise ValueError(f"{name} contains NaN/Inf in valid positions")
    norms = torch.linalg.vector_norm(valid, dim=-1)
    if bool((norms <= eps).any()):
        raise ValueError(f"{name} contains a collapsed valid vector (norm <= {eps})")
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-4):
        raise ValueError(f"{name} valid vectors must be L2-normalized")


def masked_softmax(x: Tensor, keep: Tensor, dim: int) -> Tensor:
    """Softmax over valid entries; all-invalid slices return exact zeros."""
    if keep.dtype != torch.bool:
        raise TypeError("keep mask must be bool with True=valid")
    keep = keep.expand_as(x)
    has_any = keep.any(dim=dim, keepdim=True)
    masked = x.masked_fill(~keep, float("-inf"))
    safe = torch.where(has_any, masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=dim).masked_fill(~keep, 0.0)


def evidence_score_block(
    u: Tensor,
    e: Tensor,
    video_mask: Tensor,
    text_mask: Tensor,
    tau_alignment: float = 0.07,
    *,
    validate: bool = True,
    normalize_epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Compute DIVE masked late interaction for one video/text block.

    Rows are videos and columns are texts. The expected cosine is computed in FP32,
    except FP64 is retained for numerical gradcheck.
    """
    if u.ndim != 3 or e.ndim != 3:
        raise ValueError("u and e must have shapes [B,N,D]")
    if u.shape[-1] != e.shape[-1]:
        raise ValueError("video and text feature dimensions must match")
    _require_bool_mask(video_mask, "video_mask", tuple(u.shape[:2]))
    _require_bool_mask(text_mask, "text_mask", tuple(e.shape[:2]))
    if tau_alignment <= 0:
        raise ValueError("tau_alignment must be positive")
    if validate:
        _validate_valid_vectors(u, video_mask, "video evidence", normalize_epsilon)
        _validate_valid_vectors(e, text_mask, "text units", normalize_epsilon)

    dtype = torch.float64 if u.dtype == e.dtype == torch.float64 else torch.float32
    device_type = u.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        video = torch.where(video_mask[..., None], u.to(dtype), 0.0)
        text = torch.where(text_mask[..., None], e.to(dtype), 0.0)
        interaction = torch.einsum("and,bmd->abnm", video, text)
        keep = video_mask[:, None, :, None] & text_mask[None, :, None, :]
        text_weights = masked_softmax(interaction / tau_alignment, keep, dim=-1)
        video_weights = masked_softmax(interaction / tau_alignment, keep, dim=-2)
        row = (text_weights * interaction).sum(-1).sum(-1)
        row = row / video_mask.sum(-1).clamp_min(1)[:, None]
        column = (video_weights * interaction).sum(-2).sum(-1)
        column = column / text_mask.sum(-1).clamp_min(1)[None, :]
        pair_valid = video_mask.any(-1)[:, None] & text_mask.any(-1)[None, :]
        score = torch.where(pair_valid, 0.5 * (row + column), 0.0)
    return score, pair_valid


def _slices(size: int, chunk: int) -> Iterator[slice]:
    if chunk <= 0:
        raise ValueError("chunk sizes must be positive")
    for start in range(0, size, chunk):
        yield slice(start, min(size, start + chunk))


def evidence_score_chunked(
    u: Tensor,
    e: Tensor,
    video_mask: Tensor,
    text_mask: Tensor,
    *,
    video_chunk_size: int,
    text_chunk_size: int,
    tau_alignment: float = 0.07,
    validate: bool = True,
    normalize_epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Autograd-safe block scoring with the exact dense score semantics."""
    if validate:
        if u.ndim != 3 or e.ndim != 3:
            raise ValueError("u and e must have shapes [B,N,D]")
        _require_bool_mask(video_mask, "video_mask", tuple(u.shape[:2]))
        _require_bool_mask(text_mask, "text_mask", tuple(e.shape[:2]))
        _validate_valid_vectors(u, video_mask, "video evidence", normalize_epsilon)
        _validate_valid_vectors(e, text_mask, "text units", normalize_epsilon)
    score_rows: list[Tensor] = []
    validity_rows: list[Tensor] = []
    for vs in _slices(len(u), video_chunk_size):
        scores: list[Tensor] = []
        validities: list[Tensor] = []
        for ts in _slices(len(e), text_chunk_size):
            block_score, block_valid = evidence_score_block(
                u[vs],
                e[ts],
                video_mask[vs],
                text_mask[ts],
                tau_alignment,
                validate=False,
                normalize_epsilon=normalize_epsilon,
            )
            scores.append(block_score)
            validities.append(block_valid)
        score_rows.append(torch.cat(scores, dim=1))
        validity_rows.append(torch.cat(validities, dim=1))
    return torch.cat(score_rows, dim=0), torch.cat(validity_rows, dim=0)


def compose_score(
    baseline_score: Tensor,
    student_evidence: Tensor,
    reference_evidence: Tensor,
    pair_valid: Tensor,
    gamma: float,
) -> Tensor:
    if baseline_score.shape != student_evidence.shape or baseline_score.shape != reference_evidence.shape:
        raise ValueError("baseline, student, and reference score matrices must have identical shapes")
    _require_bool_mask(pair_valid, "pair_valid", tuple(baseline_score.shape))
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    correction = torch.where(
        pair_valid,
        0.5 * (student_evidence - reference_evidence),
        torch.zeros_like(student_evidence),
    )
    return baseline_score + gamma * correction
