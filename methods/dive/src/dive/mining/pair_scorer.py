from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from dive.adapters import seds_prelogit_paired_scores

from .neighbors import NeighborError


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PersistentSedsPairScorer:
    """Exact aligned S0 scorer with chunk-level, checksum-validated resume state."""

    def __init__(
        self,
        fusion_hidden: Tensor,
        text_hidden: Tensor,
        video_mask: Tensor,
        text_mask: Tensor,
        *,
        dual_mix: float,
        temperature: float,
        device: str | torch.device,
        pair_chunk_size: int,
        cache_dir: str | Path,
        fingerprint: str,
    ) -> None:
        if (
            fusion_hidden.ndim != 3
            or text_hidden.ndim != 3
            or fusion_hidden.shape[0] != text_hidden.shape[0]
            or fusion_hidden.shape[-1] != text_hidden.shape[-1]
            or video_mask.shape != fusion_hidden.shape[:2]
            or text_mask.shape != text_hidden.shape[:2]
            or video_mask.dtype != torch.bool
            or text_mask.dtype != torch.bool
        ):
            raise NeighborError("cached S0 feature/mask shapes are incompatible")
        if pair_chunk_size <= 0 or not fingerprint:
            raise NeighborError("pair scorer chunk size/fingerprint is invalid")
        if not bool(torch.isfinite(fusion_hidden).all()) or not bool(
            torch.isfinite(text_hidden).all()
        ):
            raise NeighborError("cached S0 features contain NaN/Inf")
        if not bool(video_mask.any(dim=1).all()) or not bool(text_mask.any(dim=1).all()):
            raise NeighborError("cached S0 features contain an empty valid sequence")
        self.device = torch.device(device)
        self.fusion_hidden = fusion_hidden.to(self.device)
        self.text_hidden = text_hidden.to(self.device)
        self.video_mask = video_mask.to(self.device)
        self.text_mask = text_mask.to(self.device)
        self.dual_mix = float(dual_mix)
        self.temperature = float(temperature)
        self.pair_chunk_size = pair_chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def _identity(
        self, video_indices: np.ndarray, text_indices: np.ndarray
    ) -> tuple[str, str, str]:
        video_hash = hashlib.sha256(video_indices.astype("<i8", copy=False).tobytes()).hexdigest()
        text_hash = hashlib.sha256(text_indices.astype("<i8", copy=False).tobytes()).hexdigest()
        envelope = json.dumps(
            {
                "fingerprint": self.fingerprint,
                "video_indices_sha256": video_hash,
                "text_indices_sha256": text_hash,
                "count": len(video_indices),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(envelope).hexdigest(), video_hash, text_hash

    def __call__(self, video_indices: np.ndarray, text_indices: np.ndarray) -> np.ndarray:
        video = np.asarray(video_indices, dtype=np.int64)
        text = np.asarray(text_indices, dtype=np.int64)
        count = len(self.fusion_hidden)
        if video.ndim != 1 or text.shape != video.shape:
            raise NeighborError("pair scorer indices must be aligned vectors")
        if np.any(video < 0) or np.any(video >= count) or np.any(text < 0) or np.any(text >= count):
            raise NeighborError("pair scorer index lies outside the cached gallery")
        digest, video_hash, text_hash = self._identity(video, text)
        data_path = self.cache_dir / f"scores-{digest}.npy"
        state_path = self.cache_dir / f"scores-{digest}.json"
        expected = {
            "schema_version": "seds_sparse_pair_scores.v1",
            "fingerprint": self.fingerprint,
            "video_indices_sha256": video_hash,
            "text_indices_sha256": text_hash,
            "count": len(video),
            "dtype": "float32",
        }
        state: dict[str, Any]
        if state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise NeighborError("sparse pair-score resume state is unreadable") from exc
            if not isinstance(loaded, dict) or any(
                loaded.get(key) != value for key, value in expected.items()
            ):
                raise NeighborError("CACHE_HASH_MISMATCH: sparse pair-score resume state differs")
            completed = loaded.get("completed")
            if (
                isinstance(completed, bool)
                or not isinstance(completed, int)
                or not 0 <= completed <= len(video)
            ):
                raise NeighborError("sparse pair-score resume offset is invalid")
            state = loaded
        else:
            if data_path.exists():
                raise NeighborError("sparse pair-score data exists without resume state")
            state = expected | {"completed": 0, "scores_sha256": None}
            _atomic_json(state_path, state)
        if data_path.exists():
            try:
                scores = np.lib.format.open_memmap(data_path, mode="r+")
            except (OSError, ValueError) as exc:
                raise NeighborError("sparse pair-score data is unreadable") from exc
            if scores.shape != (len(video),) or scores.dtype != np.float32:
                raise NeighborError("sparse pair-score data shape/dtype differs")
        else:
            if state["completed"] != 0:
                raise NeighborError("sparse pair-score data is missing after recorded progress")
            scores = np.lib.format.open_memmap(
                data_path, mode="w+", dtype=np.float32, shape=(len(video),)
            )
            scores.flush()
        if state["completed"] == len(video):
            if not state.get("scores_sha256") or _sha256(data_path) != state["scores_sha256"]:
                raise NeighborError("sparse pair-score completed checksum mismatch")
            return np.asarray(scores).copy()

        with torch.no_grad():
            for start in range(int(state["completed"]), len(video), self.pair_chunk_size):
                stop = min(len(video), start + self.pair_chunk_size)
                video_index = torch.as_tensor(
                    video[start:stop], dtype=torch.long, device=self.device
                )
                text_index = torch.as_tensor(text[start:stop], dtype=torch.long, device=self.device)
                values = seds_prelogit_paired_scores(
                    self.fusion_hidden.index_select(0, video_index),
                    self.text_hidden.index_select(0, text_index),
                    self.video_mask.index_select(0, video_index),
                    self.text_mask.index_select(0, text_index),
                    dual_mix=self.dual_mix,
                    temperature=self.temperature,
                )[0]
                scores[start:stop] = values.float().cpu().numpy()
                scores.flush()
                state = expected | {"completed": stop, "scores_sha256": None}
                _atomic_json(state_path, state)
        state["scores_sha256"] = _sha256(data_path)
        _atomic_json(state_path, state)
        return np.asarray(scores).copy()
