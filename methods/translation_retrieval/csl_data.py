"""Deterministic CSL-Daily 133-keypoint input for pose-only retrieval.

The normalization follows the public Uni-Sign pose-input convention, but this
loader is independent of its training framework and never reads TEST labels.
"""

from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


PART_INDICES = {
    "body": np.asarray([0, *range(3, 11)]),
    "left": np.arange(91, 112),
    "right": np.arange(112, 133),
    "face_all": np.asarray([*range(23, 40, 2), *range(83, 91), 53]),
}
CONFIDENCE_THRESHOLD = 0.3


def select_frames(length: int, limit: int, *, name: str, seed: int, epoch: int,
                  train: bool) -> np.ndarray:
    if length < 1 or limit < 1:
        raise ValueError("length and limit must be positive")
    if length <= limit:
        return np.arange(length)
    if not train:
        return np.linspace(0, length - 1, limit, dtype=np.int64)
    digest = hashlib.sha256(f"{seed}:{epoch}:{name}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return np.sort(generator.choice(length, size=limit, replace=False))


def split_pose_parts(keypoints: np.ndarray, scores: np.ndarray) -> dict[str, Tensor]:
    """Map [T,133,2] and [T,133] detector output to Uni-Sign's four parts."""
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2):
        raise ValueError("keypoints must be [T,133,2]")
    if scores.shape != keypoints.shape[:2]:
        raise ValueError("scores must be [T,133]")
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        raise ValueError("non-finite pose values")

    # The body box is computed across all valid body joints in the clip. Every
    # other part is centered at its own anchor but shares that body scale.
    body_ids = PART_INDICES["body"]
    body_xy = keypoints[:, body_ids]
    body_score = scores[:, body_ids]
    valid = body_score > CONFIDENCE_THRESHOLD
    valid_xy = body_xy[valid]
    if len(valid_xy) < 4:
        return {
            name: torch.zeros(keypoints.shape[0], len(ids), 3, dtype=torch.float32)
            for name, ids in PART_INDICES.items()
        }
    minimum = valid_xy.min(axis=0)
    maximum = valid_xy.max(axis=0)
    scale = float((maximum - minimum).max())
    if scale <= 0:
        return {
            name: torch.zeros(keypoints.shape[0], len(ids), 3, dtype=torch.float32)
            for name, ids in PART_INDICES.items()
        }

    result: dict[str, Tensor] = {}
    for name, ids in PART_INDICES.items():
        xy = keypoints[:, ids].astype(np.float32, copy=True)
        confidence = scores[:, ids].astype(np.float32, copy=True)
        if name == "body":
            center = (minimum + maximum) / 2
            xy = 2 * (xy - center) / scale
        else:
            anchor = xy[:, -1:, :] if name == "face_all" else xy[:, :1, :]
            xy = (xy - anchor) / scale
        packed = np.concatenate([xy, confidence[..., None]], axis=-1).astype(
            np.float32, copy=False
        )
        np.clip(packed, -1, 1, out=packed)
        packed[confidence <= CONFIDENCE_THRESHOLD] = 0
        result[name] = torch.from_numpy(packed)
    return result


class UniSignCSLPoseDataset(Dataset):
    """Reads the local CSL labels and 133-point pose files without extraction."""

    def __init__(self, labels_path: str | Path, poses_dir: str | Path, *,
                 split: str, max_length: int = 256, seed: int = 42) -> None:
        if split not in {"train", "dev"}:
            raise ValueError("only TRAIN and DEV are admitted during selection")
        self.split = split
        self.poses_dir = Path(poses_dir)
        self.max_length = max_length
        self.seed = seed
        self.epoch = 0
        with gzip.open(labels_path, "rb") as stream:
            labels: Any = pickle.load(stream)
        if not isinstance(labels, dict):
            raise ValueError("expected a dictionary of Uni-Sign CSL labels")
        self.records = [labels[name] for name in sorted(labels)]

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[str, dict[str, Tensor], str]:
        record = self.records[index]
        name = str(record["name"])
        video_path = Path(str(record["video_path"]))
        if video_path.name != str(video_path) or video_path.suffix != ".mp4":
            raise ValueError("unsafe or unexpected video path in labels")
        with (self.poses_dir / video_path.with_suffix(".pkl")).open("rb") as stream:
            pose: Any = pickle.load(stream)
        points = pose["keypoints"]
        confidence = pose["scores"]
        if len(points) != len(confidence):
            raise ValueError(f"pose/score frame mismatch: {name}")
        indices = select_frames(len(points), self.max_length, name=name,
                                seed=self.seed, epoch=self.epoch,
                                train=self.split == "train")
        xy = np.stack([np.asarray(points[i])[0] for i in indices])
        score = np.stack([np.asarray(confidence[i])[0] for i in indices])
        return name, split_pose_parts(xy, score), str(record["text"])


def collate_pose_batch(
    examples: list[tuple[str, dict[str, Tensor], str]],
) -> tuple[list[str], dict[str, Tensor], Tensor, list[str]]:
    if not examples:
        raise ValueError("empty batch")
    lengths = [examples[i][1]["body"].shape[0] for i in range(len(examples))]
    max_len = max(lengths)
    mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    padded: dict[str, Tensor] = {}
    for part in PART_INDICES:
        clips = [item[1][part] for item in examples]
        padded[part] = torch.stack([
            torch.cat([clip, clip[-1:].expand(max_len - len(clip), -1, -1)], dim=0)
            for clip in clips
        ])
    return [item[0] for item in examples], padded, mask.long(), [item[2] for item in examples]
