from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from torch import Tensor, nn


@dataclass(frozen=True)
class NativeVideoFeatures:
    sample_ids: tuple[str, ...]
    pooled: Tensor
    validity: Tensor
    streams: Mapping[str, Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeTextFeatures:
    text_ids: tuple[str, ...]
    pooled: Tensor
    token_features: Tensor
    token_validity: Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrelogitScores:
    video_ids: tuple[str, ...]
    text_ids: tuple[str, ...]
    scores: Tensor
    logit_scale: Tensor
    directional_scores: Mapping[str, Tensor] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaselineAdapter(Protocol):
    """Side-effect-free contract between DIVE and a locked retrieval baseline."""

    def load_and_validate(
        self, checkpoint_path: Path, resolved_config: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def encode_video_native(self, video_batch: Any) -> NativeVideoFeatures: ...

    def encode_text_native(self, text_batch: Any) -> NativeTextFeatures: ...

    def score_prelogit(
        self, video_features: NativeVideoFeatures, text_features: NativeTextFeatures
    ) -> PrelogitScores: ...

    def encode_text_units(self, text_batch: Any, unit_mapping: Any) -> NativeTextFeatures: ...

    def rgb_local_features(self, video_batch: Any, grid_id: str) -> NativeVideoFeatures: ...

    def clone_local_pose_encoder(self) -> nn.Module: ...

    def describe_preprocessing(self) -> Mapping[str, Any]: ...

    def describe_receptive_field(
        self, video_batch: Any, grid_id: str
    ) -> Sequence[Mapping[str, Any]]: ...
