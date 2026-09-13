from __future__ import annotations

import pickle
import random
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Method1Config
from .manifests import iter_jsonl
from .sampling import choose_group_member, mix_and_sample_features
from .schemas import GroupRecord, TextRecord, VideoRecord
from .token_spans import tokenize_with_spans
from .utils import stable_seed


class DataError(ValueError):
    pass


def _load_records(path: Path, cls):
    allowed = {item.name for item in fields(cls)}
    records = []
    for value in iter_jsonl(path):
        unknown = set(value) - allowed
        if unknown:
            raise DataError(f"unknown fields in {path.name}: {sorted(unknown)}")
        if cls is GroupRecord:
            value["video_uids"] = tuple(value["video_uids"])
        record = cls(**value)
        record.validate()
        records.append(record)
    return records


def _load_feature(path: str) -> np.ndarray:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - path comes from audited trusted manifest
    if isinstance(value, dict):
        value = value.get("feature")
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 1024 or not np.isfinite(array).all():
        raise DataError(f"invalid feature array at {path}: {array.shape}")
    return array


def _random_swap(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    left = rng.randint(0, len(words) - 1)
    right = left
    counter = 0
    while right == left:
        right = rng.randint(0, len(words) - 1)
        counter += 1
        if counter > 3:
            return " ".join(words)
    words[left], words[right] = words[right], words[left]
    return " ".join(words)


def augment_caption(
    text: str,
    *,
    seed: int,
    epoch: int,
    text_uid: str,
    probability: float = 0.5,
) -> str:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation probability must be in [0,1]")
    rng = random.Random(stable_seed(seed, epoch, text_uid, "baseline_text_augmentation"))
    return _random_swap(text, rng) if rng.random() > 1.0 - probability else text


class Method1Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        config: Method1Config,
        *,
        split: str,
        tokenizer: Any,
        augment: bool,
    ) -> None:
        if split not in {"train", "dev", "test"}:
            raise DataError(f"unsupported split: {split}")
        root = Path(config.data.manifest_dir)
        texts = _load_records(root / "texts.jsonl", TextRecord)
        videos = _load_records(root / "videos.jsonl", VideoRecord)
        groups = _load_records(root / "groups.jsonl", GroupRecord)
        self.texts = {record.text_uid: record for record in texts if record.split == split}
        self.videos = {record.video_uid: record for record in videos if record.split == split}
        self.groups = sorted(
            (record for record in groups if record.group_uid.startswith(f"{config.data.dataset}:{split}:")),
            key=lambda record: record.official_order,
        )
        if not self.groups:
            raise DataError(f"no manifest groups for split={split}")
        for group in self.groups:
            if group.text_uid not in self.texts:
                raise DataError(f"group references missing text: {group.group_uid}")
            if any(video_uid not in self.videos for video_uid in group.video_uids):
                raise DataError(f"group references missing video: {group.group_uid}")
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.augment = bool(augment)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        if self.split == "train":
            video_uid = choose_group_member(
                group.video_uids,
                seed=self.config.seed,
                epoch=self.epoch,
                group_uid=group.group_uid,
            )
        else:
            # Evaluation enumerates actual videos through an explicit flattened view; this
            # group-sampled dataset is for base training and per-group text encoding only.
            video_uid = group.video_uids[0]
        video = self.videos[video_uid]
        text = self.texts[group.text_uid]
        agnostic = _load_feature(video.agnostic_path)
        aware = _load_feature(video.aware_path)
        features, valid, selected = mix_and_sample_features(
            agnostic,
            aware,
            agnostic_weight=self.config.data.agnostic_weight,
            feature_len=self.config.data.feature_len,
        )
        encoded = tokenize_with_spans(
            text.raw_text,
            text_uid=text.text_uid,
            tokenizer=self.tokenizer,
            max_positions=self.config.data.text_max_positions,
        )
        augmented_text = (
            augment_caption(
                text.raw_text,
                seed=self.config.seed,
                epoch=self.epoch,
                text_uid=text.text_uid,
                probability=self.config.data.baseline_text_augmentation_probability,
            )
            if self.augment
            else text.raw_text
        )
        augmented = tokenize_with_spans(
            augmented_text,
            text_uid=text.text_uid,
            tokenizer=self.tokenizer,
            max_positions=self.config.data.text_max_positions,
        )
        video_ignore = np.ones(self.config.data.feature_len + 1, dtype=np.bool_)
        video_ignore[1:] = ~valid
        return {
            "video_uid": video_uid,
            "text_uid": text.text_uid,
            "group_uid": group.group_uid,
            "caption_hash": text.caption_hash,
            "video_features": torch.from_numpy(features.T[:, :, None].copy()),
            "video_ignore_raw": torch.from_numpy(video_ignore),
            "selected_feature_indices": torch.from_numpy(selected.copy()),
            "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
            "text_valid": torch.tensor(encoded.text_valid, dtype=torch.bool),
            "input_ids_aug": torch.tensor(augmented.input_ids, dtype=torch.long),
            "text_aug_valid": torch.tensor(augmented.text_valid, dtype=torch.bool),
            "token_type_ids": torch.zeros(self.config.data.text_max_positions, dtype=torch.long),
            "canonical_text": text.canonical_text,
            "lexical_spans": encoded.lexical_spans,
        }


class Method1Collator:
    _TENSOR_KEYS = (
        "video_features",
        "video_ignore_raw",
        "selected_feature_indices",
        "input_ids",
        "text_valid",
        "input_ids_aug",
        "text_aug_valid",
        "token_type_ids",
    )

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise DataError("cannot collate an empty batch")
        output = {
            key: [item[key] for item in items]
            for key in (
                "video_uid",
                "text_uid",
                "group_uid",
                "caption_hash",
                "canonical_text",
                "lexical_spans",
            )
        }
        for key in self._TENSOR_KEYS:
            output[key] = torch.stack([item[key] for item in items])
        if output["video_features"].ndim != 4 or output["video_features"].shape[1:] != (
            1024,
            64,
            1,
        ):
            raise DataError("collated video_features violate [B,1024,64,1]")
        return output


class TrimmedDistributedGroupSampler(Sampler[int]):
    """One unique group per epoch, trimmed before rank sharding to avoid duplicates."""

    def __init__(
        self,
        length: int,
        *,
        seed: int,
        global_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if length < 1 or global_batch_size < 1 or world_size < 1:
            raise ValueError("length, global_batch_size, and world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0,world_size)")
        if global_batch_size % world_size:
            raise ValueError("global batch size must be divisible by world size")
        self.length = length
        self.seed = seed
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.usable = (length // global_batch_size) * global_batch_size
        if not self.usable:
            raise ValueError("dataset is smaller than one global contrastive batch")

    @property
    def dropped_tail(self) -> int:
        return self.length - self.usable

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.usable // self.world_size

    def __iter__(self):
        generator = np.random.default_rng(
            stable_seed(self.seed, self.epoch, "trimmed_group_permutation")
        )
        permutation = generator.permutation(self.length)[: self.usable]
        return iter(int(index) for index in permutation[self.rank :: self.world_size])
