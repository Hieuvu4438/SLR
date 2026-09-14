from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from pmgr.data.group_index import CanonicalGroupIndex, GroupRecord, load_group_index
from slr_common.data.cico_dataset import FeatureFormatError, load_dense_feature
from slr_common.data.tokenize import encode_cico_text, random_swap_words
from slr_common.data.views import canonical_view, fuse_features, materialize_view
from slr_common.utils import stable_seed


class GroupDataError(ValueError):
    pass


class GroupDataset(Dataset[dict[str, Any]]):
    def __init__(self, index: str | CanonicalGroupIndex, *, feature_len: int, alpha: float):
        self.index = load_group_index(index) if isinstance(index, str) else index
        if feature_len < 1 or not 0.0 <= alpha <= 1.0:
            raise GroupDataError("invalid feature packing settings")
        self.feature_len = int(feature_len)
        self.alpha = float(alpha)

    def __len__(self) -> int:
        return self.index.group_count

    def _load_video(self, group: GroupRecord, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        video = group.videos[position]
        try:
            original = load_dense_feature(video.original_feature)
            adapted = load_dense_feature(video.adapted_feature)
        except (OSError, FeatureFormatError) as error:
            raise GroupDataError(f"cannot load {video.video_id}: {error}") from error
        expected = (video.dense_length, video.feature_dim)
        if original.shape != expected or adapted.shape != expected:
            raise GroupDataError(
                f"{video.video_id}: feature streams must both have shape {expected}, "
                f"got {tuple(original.shape)} and {tuple(adapted.shape)}"
            )
        if not bool(torch.isfinite(original).all()) or not bool(torch.isfinite(adapted).all()):
            raise GroupDataError(f"{video.video_id}: feature contains NaN or infinity")
        dense = fuse_features(adapted, original, alpha=self.alpha)
        packed, valid, _ = materialize_view(
            dense, canonical_view(video.dense_length, self.feature_len), self.feature_len
        )
        return packed, valid

    def __getitem__(self, group_index: int) -> dict[str, Any]:
        group = self.index.groups[group_index]
        loaded = [self._load_video(group, position) for position in range(len(group.videos))]
        return {
            "group_id": group.group_id,
            "canonical_text": group.canonical_text,
            "original_text": group.original_text,
            "video_ids": [video.video_id for video in group.videos],
            "full_group_size": len(group.videos),
            "features": torch.stack([item[0] for item in loaded]),
            "valid": torch.stack([item[1] for item in loaded]),
        }


class GroupCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_words: int,
        *,
        dataset_group_count: int,
        dataset_video_count: int,
        seed: int,
        augment: bool = True,
        augmentation_probability: float = 0.5,
        representative_policy: str = "all",
    ):
        if max_words < 2 or not 0.0 <= augmentation_probability <= 1.0:
            raise GroupDataError("invalid text packing settings")
        if representative_policy not in {"all", "uniform_deterministic"}:
            raise GroupDataError("unsupported representative policy")
        self.tokenizer = tokenizer
        self.max_words = int(max_words)
        self.dataset_group_count = int(dataset_group_count)
        self.dataset_video_count = int(dataset_video_count)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.augmentation_probability = float(augmentation_probability)
        self.representative_policy = representative_policy
        self.epoch = 0
        self.step = 0

    def set_position(self, epoch: int, step: int) -> None:
        if epoch < 0 or step < 0:
            raise GroupDataError("epoch and effective step must be non-negative")
        self.epoch, self.step = int(epoch), int(step)

    def _augmented(self, text: str, group_id: str) -> str:
        if not self.augment:
            return text
        rng = random.Random(
            stable_seed(self.seed, self.epoch, self.step, group_id, "cico_random_swap_v1")
        )
        return random_swap_words(text, rng) if rng.random() < self.augmentation_probability else text

    def _select(self, item: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if self.representative_policy == "all":
            if len(item["video_ids"]) != int(item["full_group_size"]):
                raise GroupDataError(f"group {item['group_id']} was truncated")
            return item["features"], item["valid"], list(item["video_ids"])
        rng = random.Random(
            stable_seed(self.seed, self.epoch, self.step, item["group_id"], "representative")
        )
        position = rng.randrange(int(item["full_group_size"]))
        return (
            item["features"][position : position + 1],
            item["valid"][position : position + 1],
            [item["video_ids"][position]],
        )

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if len(items) < 2:
            raise GroupDataError("PMGR training requires at least two distinct groups")
        group_ids = [str(item["group_id"]) for item in items]
        if len(group_ids) != len(set(group_ids)):
            raise GroupDataError("group batch contains duplicate groups")
        selected = [self._select(item) for item in items]
        selected_sizes = torch.tensor([len(value[2]) for value in selected], dtype=torch.long)
        full_sizes = torch.tensor([int(item["full_group_size"]) for item in items], dtype=torch.long)
        owner = torch.repeat_interleave(torch.arange(len(items)), selected_sizes)
        if not torch.equal(torch.bincount(owner, minlength=len(items)), selected_sizes):
            raise GroupDataError("video-to-group mapping does not match selected group sizes")
        features = torch.cat([value[0] for value in selected])
        valid = torch.cat([value[1] for value in selected])
        video_padding_mask = torch.ones((len(features), features.shape[1] + 1), dtype=torch.long)
        video_padding_mask[:, 1:] = (~valid).long()
        clean = [encode_cico_text(item["canonical_text"], self.tokenizer, self.max_words) for item in items]
        augmented_strings = [
            self._augmented(item["canonical_text"], item["group_id"]) for item in items
        ]
        augmented = [encode_cico_text(text, self.tokenizer, self.max_words) for text in augmented_strings]
        output: dict[str, Any] = {
            "group_ids": group_ids,
            "video_ids": [identifier for value in selected for identifier in value[2]],
            "video_to_group": owner,
            "selected_group_sizes": selected_sizes,
            "full_group_sizes": full_sizes,
            "video_features": features.transpose(1, 2).unsqueeze(-1).contiguous(),
            "video_padding_mask": video_padding_mask,
            "dataset_group_count": self.dataset_group_count,
            "dataset_video_count": self.dataset_video_count,
            "augmented_text": augmented_strings,
        }
        names = ("input_ids", "segment_ids", "input_mask")
        aug_names = ("aug_input_ids", "aug_segment_ids", "aug_input_mask")
        for position, name in enumerate(names):
            output[name] = torch.stack([value[position] for value in clean])
        for position, name in enumerate(aug_names):
            output[name] = torch.stack([value[position] for value in augmented])
        return output
