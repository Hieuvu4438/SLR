from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class FixtureContrast:
    pair_id: str
    sample_i: str
    sample_j: str
    unit_i: int
    unit_j: int
    support_status: str
    reliability: float


@dataclass(frozen=True)
class SyntheticFixture:
    """Small planted association used only for contract and gradient tests."""

    sample_ids: tuple[str, ...]
    video_ids: tuple[str, ...]
    text_ids: tuple[str, ...]
    video_evidence: Tensor
    text_units: Tensor
    video_mask: Tensor
    text_mask: Tensor
    baseline_scores: Tensor
    relevance: dict[str, frozenset[str]]
    contrasts: tuple[FixtureContrast, ...]
    excluded_negatives: frozenset[tuple[str, str]]
    tie_pair: tuple[tuple[str, str], tuple[str, str]]


def build_synthetic_fixture(seed: int = 17) -> SyntheticFixture:
    """Build a deterministic fixture with every §19.1 behavioral case.

    Associations are planted directly in normalized feature directions. This fixture says nothing
    about sign grounding or benchmark performance.
    """
    generator = torch.Generator().manual_seed(seed)
    count, max_clips, max_units, dimension = 4, 5, 4, 8
    concepts = F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)
    video = F.normalize(
        concepts[:, None, :] + 0.04 * torch.randn(count, max_clips, dimension, generator=generator),
        dim=-1,
    )
    text = F.normalize(
        concepts[:, None, :] + 0.04 * torch.randn(count, max_units, dimension, generator=generator),
        dim=-1,
    )
    video_lengths = torch.tensor([5, 4, 3, 2])
    text_lengths = torch.tensor([4, 3, 2, 1])
    video_mask = torch.arange(max_clips)[None, :] < video_lengths[:, None]
    text_mask = torch.arange(max_units)[None, :] < text_lengths[:, None]
    video = torch.where(video_mask[..., None], video, torch.zeros_like(video))
    text = torch.where(text_mask[..., None], text, torch.zeros_like(text))
    baseline = concepts @ concepts.T
    # An exact tie exercises deterministic secondary-ID ranking.
    baseline[3, 1] = baseline[3, 2]
    sample_ids = tuple(f"fixture_train_{index:04d}" for index in range(count))
    video_ids = tuple(f"fixture_video_{index:04d}" for index in range(count))
    text_ids = tuple(f"fixture_text_{index:04d}" for index in range(count))
    relevance = {video_id: frozenset({text_id}) for video_id, text_id in zip(video_ids, text_ids, strict=True)}
    return SyntheticFixture(
        sample_ids=sample_ids,
        video_ids=video_ids,
        text_ids=text_ids,
        video_evidence=video,
        text_units=text,
        video_mask=video_mask,
        text_mask=text_mask,
        baseline_scores=baseline,
        relevance=relevance,
        contrasts=(
            FixtureContrast("fixture_pair_accepted", sample_ids[0], sample_ids[1], 0, 0, "accepted", 1.0),
            FixtureContrast("fixture_pair_failed", sample_ids[2], sample_ids[3], 0, 0, "tiny_text_distance", 0.0),
        ),
        excluded_negatives=frozenset({(video_ids[0], text_ids[2])}),
        tie_pair=((video_ids[3], text_ids[1]), (video_ids[3], text_ids[2])),
    )
