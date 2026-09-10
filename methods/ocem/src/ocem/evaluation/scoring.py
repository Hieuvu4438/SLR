"""Caption-isolated block scoring over complete video and text indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


class FullGalleryScoringError(ValueError):
    """Raised when a scorer omits candidates or violates index contracts."""


@dataclass(frozen=True)
class VideoIndex:
    """Inference-only video state; candidate captions cannot be stored here."""

    ids: tuple[str, ...]
    representations: torch.Tensor
    valid: torch.Tensor
    intervals: torch.Tensor

    def __post_init__(self) -> None:
        size = len(self.ids)
        if size == 0 or len(set(self.ids)) != size:
            raise FullGalleryScoringError("video index IDs must be non-empty and unique")
        if self.representations.shape[0] != size or self.valid.shape[0] != size:
            raise FullGalleryScoringError("video representations/masks do not match IDs")
        if self.intervals.shape[:2] != self.valid.shape or self.intervals.shape[-1] != 2:
            raise FullGalleryScoringError("video intervals must align with the validity mask")
        if self.valid.dtype is not torch.bool:
            raise FullGalleryScoringError("video validity mask must have bool dtype")


@dataclass(frozen=True)
class TextIndex:
    """Ordinary text query/gallery encodings, independent of target labels."""

    ids: tuple[str, ...]
    representations: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        size = len(self.ids)
        if size == 0 or len(set(self.ids)) != size:
            raise FullGalleryScoringError("text index IDs must be non-empty and unique")
        if self.representations.shape[0] != size or self.valid.shape[0] != size:
            raise FullGalleryScoringError("text representations/masks do not match IDs")
        if self.valid.dtype is not torch.bool:
            raise FullGalleryScoringError("text validity mask must have bool dtype")


@dataclass(frozen=True)
class DirectionalScoreMatrices:
    scores_t2v: torch.Tensor
    scores_v2t: torch.Tensor
    scored_pairs: int
    expected_pairs: int


BlockScorer = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


def score_directional_full_gallery(
    video_index: VideoIndex,
    text_index: TextIndex,
    scorer: BlockScorer,
    *,
    video_block_size: int,
    text_block_size: int,
    output_device: torch.device | str = "cpu",
) -> DirectionalScoreMatrices:
    """Score every Cartesian pair; target mappings belong only to the evaluator."""

    if video_block_size < 1 or text_block_size < 1:
        raise FullGalleryScoringError("block sizes must be positive")
    video_count, text_count = len(video_index.ids), len(text_index.ids)
    t2v = torch.full(
        (video_count, text_count), torch.nan, dtype=torch.float32, device=output_device
    )
    v2t = torch.full_like(t2v, torch.nan)
    scored_pairs = 0
    for video_start in range(0, video_count, video_block_size):
        video_stop = min(video_start + video_block_size, video_count)
        for text_start in range(0, text_count, text_block_size):
            text_stop = min(text_start + text_block_size, text_count)
            block_t2v, block_v2t = scorer(
                video_index.representations[video_start:video_stop],
                video_index.valid[video_start:video_stop],
                video_index.intervals[video_start:video_stop],
                text_index.representations[text_start:text_stop],
                text_index.valid[text_start:text_stop],
            )
            expected_shape = (video_stop - video_start, text_stop - text_start)
            if block_t2v.shape != expected_shape or block_v2t.shape != expected_shape:
                raise FullGalleryScoringError(
                    f"scorer blocks must have shape {expected_shape}, got "
                    f"{tuple(block_t2v.shape)} and {tuple(block_v2t.shape)}"
                )
            if not torch.all(torch.isfinite(block_t2v)) or not torch.all(
                torch.isfinite(block_v2t)
            ):
                raise FullGalleryScoringError("scorer returned a non-finite value")
            t2v[video_start:video_stop, text_start:text_stop] = block_t2v.detach().to(
                device=output_device, dtype=torch.float32
            )
            v2t[video_start:video_stop, text_start:text_stop] = block_v2t.detach().to(
                device=output_device, dtype=torch.float32
            )
            scored_pairs += expected_shape[0] * expected_shape[1]
    expected_pairs = video_count * text_count
    if scored_pairs != expected_pairs or torch.any(torch.isnan(t2v)) or torch.any(torch.isnan(v2t)):
        raise FullGalleryScoringError("full-gallery coverage is incomplete")
    return DirectionalScoreMatrices(
        scores_t2v=t2v,
        scores_v2t=v2t,
        scored_pairs=scored_pairs,
        expected_pairs=expected_pairs,
    )


def build_video_index(
    ids: Sequence[str],
    representations: torch.Tensor,
    valid: torch.Tensor,
    intervals: torch.Tensor,
) -> VideoIndex:
    """Build a scorer index from visual inputs only (no annotation argument)."""

    return VideoIndex(tuple(map(str, ids)), representations, valid, intervals)


def build_text_index(
    ids: Sequence[str], representations: torch.Tensor, valid: torch.Tensor
) -> TextIndex:
    """Build the ordinary text query/gallery index without target mappings."""

    return TextIndex(tuple(map(str, ids)), representations, valid)
