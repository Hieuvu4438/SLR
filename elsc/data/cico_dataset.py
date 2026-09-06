from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from elsc.data.manifest import ManifestRecord, load_manifest
from elsc.data.views import canonical_view, fuse_features, jittered_view, materialize_view


class FeatureFormatError(ValueError):
    pass


def load_dense_feature(path: str | Path) -> torch.Tensor:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict):
        value = value.get("feature")
    array = np.asarray(value)
    if array.ndim != 2:
        raise FeatureFormatError(f"expected a rank-2 feature array at {path}, got {array.shape}")
    # CiCo archives store [D,L]; accept [L,D] where the manifest confirms it later.
    if array.shape[0] == 1024 and array.shape[1] != 1024:
        array = array.T
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


class CiCoFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        *,
        feature_len: int,
        alpha: float,
        split: str,
        include_jittered_view: bool = False,
        seed: int = 42,
    ):
        self.records = load_manifest(manifest, expected_split=split)
        self.feature_len = int(feature_len)
        self.alpha = float(alpha)
        self.split = split
        self.include_jittered_view = include_jittered_view
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record: ManifestRecord = self.records[index]
        aware = load_dense_feature(record.feature_aware)
        agnostic = load_dense_feature(record.feature_agnostic)
        if aware.shape != (record.dense_length, record.feature_dim):
            raise FeatureFormatError(
                f"{record.pair_id}: expected {(record.dense_length, record.feature_dim)}, got {aware.shape}"
            )
        dense_h = fuse_features(aware, agnostic, alpha=self.alpha)
        view = canonical_view(record.dense_length, self.feature_len)
        h, valid, dense_index = materialize_view(dense_h, view, self.feature_len)
        output = {
            "pair_id": record.pair_id,
            "video_id": record.video_id,
            "caption_id": record.caption_id,
            "caption": record.caption_model,
            "split": record.split,
            "h": h,
            "valid": valid,
            "dense_index": dense_index,
            "view_hash": view.hash,
        }
        if self.include_jittered_view:
            view_b, independent = jittered_view(
                record.dense_length, self.feature_len, global_seed=self.seed, video_id=record.video_id
            )
            h_b, valid_b, dense_index_b = materialize_view(dense_h, view_b, self.feature_len)
            output.update(
                h_b=h_b,
                valid_b=valid_b,
                dense_index_b=dense_index_b,
                view_b_hash=view_b.hash,
                view_independent=independent,
            )
        return output
