from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from slr_common.utils import stable_seed


@dataclass(frozen=True)
class TemporalView:
    indices: np.ndarray
    dense_length: int
    name: str

    @property
    def hash(self) -> str:
        payload = f"v1:{self.name}:{self.dense_length}:" + ",".join(map(str, self.indices.tolist()))
        return hashlib.sha256(payload.encode()).hexdigest()


def uniform_indices(dense_length: int, feature_len: int) -> np.ndarray:
    if dense_length <= 0 or feature_len <= 0:
        raise ValueError("dense_length and feature_len must be positive")
    if dense_length >= feature_len:
        return np.linspace(0, dense_length - 1, feature_len, dtype=np.int64)
    return np.arange(dense_length, dtype=np.int64)


def canonical_view(dense_length: int, feature_len: int) -> TemporalView:
    return TemporalView(uniform_indices(dense_length, feature_len), dense_length, "view_a")


def jittered_view(
    dense_length: int, feature_len: int, *, global_seed: int, video_id: str
) -> tuple[TemporalView, bool]:
    base = uniform_indices(dense_length, feature_len)
    if dense_length <= 1 or len(base) >= dense_length:
        return TemporalView(base.copy(), dense_length, "view_b"), False
    generator = np.random.default_rng(stable_seed(global_seed, video_id, "view_b"))
    candidates: list[list[int]] = []
    for index in base:
        candidates.append(
            [value for value in (int(index) - 1, int(index), int(index) + 1) if 0 <= value < dense_length]
        )
    for _ in range(16):
        proposed = np.array([generator.choice(values) for values in candidates], dtype=np.int64)
        proposed.sort()
        if len(np.unique(proposed)) == len(proposed) and not np.array_equal(proposed, base):
            return TemporalView(proposed, dense_length, "view_b"), True
    available = [idx for idx in range(dense_length) if idx not in set(base.tolist())]
    for replacement in available:
        distances = np.abs(base - replacement)
        position = int(np.argmin(distances))
        if distances[position] <= 1:
            proposed = base.copy()
            proposed[position] = replacement
            proposed.sort()
            if len(np.unique(proposed)) == len(proposed):
                return TemporalView(proposed, dense_length, "view_b"), True
    return TemporalView(base.copy(), dense_length, "view_b"), False


def materialize_view(dense_h: torch.Tensor, view: TemporalView, feature_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dense_h.ndim != 2 or dense_h.shape[0] != view.dense_length:
        raise ValueError("dense_h must be [dense_length, feature_dim]")
    h = dense_h.new_zeros((feature_len, dense_h.shape[1]))
    valid = torch.zeros(feature_len, dtype=torch.bool, device=dense_h.device)
    dense_index = torch.full((feature_len,), -1, dtype=torch.long, device=dense_h.device)
    count = min(feature_len, len(view.indices))
    chosen = torch.as_tensor(view.indices[:count], dtype=torch.long, device=dense_h.device)
    h[:count] = dense_h.index_select(0, chosen)
    valid[:count] = True
    dense_index[:count] = chosen
    return h, valid, dense_index


def fuse_features(aware: torch.Tensor, agnostic: torch.Tensor, *, alpha: float) -> torch.Tensor:
    if aware.shape != agnostic.shape:
        raise ValueError(f"feature streams differ: {aware.shape} != {agnostic.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    return (1.0 - alpha) * aware + alpha * agnostic
