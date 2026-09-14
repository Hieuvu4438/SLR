from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from pmgr.data.group_dataset import GroupCollator, GroupDataError, GroupDataset
from pmgr.data.group_index import build_group_index, load_group_index
from pmgr.data.group_sampler import GroupBatchSampler
from slr_common.data.manifest import ManifestRecord, write_manifest


class TinyTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return list(range(1, len(tokens) + 1))


def _write_feature(path: Path, value: float, length: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(np.full((length, 1024), value, dtype=np.float32), handle)


def _index(tmp_path: Path):
    records = []
    specs = [("v0", "g0", "same text"), ("v1", "g0", "same text"),
             ("v2", "g1", "same text"), ("v3", "g2", "other text"),
             ("v4", "g2", "other text"), ("v5", "g2", "other text")]
    for position, (video, group, text) in enumerate(specs):
        original = tmp_path / "original" / f"{video}.pkl"
        adapted = tmp_path / "adapted" / f"{video}.pkl"
        _write_feature(original, position + 1)
        _write_feature(adapted, position + 2)
        records.append(
            ManifestRecord(1, "fixture", "train", video, video, group, text, text, "en",
                           str(original), str(adapted), 3, 1024)
        )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    index_path = tmp_path / "index.json"
    report = build_group_index(manifest, index_path, expected_split="train")
    return index_path, report


def test_build_index_keeps_same_text_groups_distinct(tmp_path):
    path, report = _index(tmp_path)
    index = load_group_index(path, check_paths=True)
    assert report["groups"] == 3 and report["videos"] == 6
    assert [group.group_id for group in index.groups] == ["g0", "g1", "g2"]
    assert [len(group.videos) for group in index.groups] == [2, 1, 3]
    assert index.groups[0].canonical_text == index.groups[1].canonical_text


def test_dataset_and_collator_flatten_complete_groups(tmp_path):
    path, _ = _index(tmp_path)
    dataset = GroupDataset(str(path), feature_len=4, alpha=0.8)
    collator = GroupCollator(TinyTokenizer(), 8, dataset_group_count=3,
                             dataset_video_count=6, seed=2, augment=True)
    batch = collator([dataset[index] for index in range(3)])
    assert batch["video_features"].shape == (6, 1024, 4, 1)
    assert batch["video_padding_mask"].shape == (6, 5)
    assert torch.equal(batch["video_to_group"], torch.tensor([0, 0, 1, 2, 2, 2]))
    assert torch.equal(batch["selected_group_sizes"], torch.tensor([2, 1, 3]))
    assert batch["input_ids"].shape == (3, 8)
    assert len(batch["augmented_text"]) == 3


def test_deliberately_truncated_group_fails(tmp_path):
    path, _ = _index(tmp_path)
    dataset = GroupDataset(str(path), feature_len=4, alpha=0.8)
    item0, item1 = dataset[0], dataset[1]
    item0["video_ids"] = item0["video_ids"][:1]
    item0["features"] = item0["features"][:1]
    item0["valid"] = item0["valid"][:1]
    collator = GroupCollator(TinyTokenizer(), 8, dataset_group_count=3,
                             dataset_video_count=6, seed=2)
    with pytest.raises(GroupDataError, match="truncated"):
        collator([item0, item1])


def test_sampler_is_deterministic_and_never_pads_tail():
    sampler = GroupBatchSampler(7, 3, seed=9, drop_last=True)
    first = list(sampler)
    second = list(sampler)
    assert first == second and len(first) == 2
    flattened = [value for batch in first for value in batch]
    assert len(flattened) == len(set(flattened)) == 6
    assert len(sampler.dropped_indices) == 1
