from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path

from elsc.transfer_audit import audit_csl_daily, audit_how2sign


def _touch_videos(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True)
    for name in names:
        (root / name).write_bytes(b"video")


def test_how2sign_audit_preserves_clip_ids_and_detects_missing_video(tmp_path: Path):
    root = tmp_path / "h2s"
    train_annotation = root / "train" / "train_label" / "labels.train"
    train_annotation.parent.mkdir(parents=True)
    train_annotation.write_text(
        json.dumps({"video_path": "train.mp4", "text": "train text"}) + "\n",
        encoding="utf-8",
    )
    _touch_videos(root / "train" / "raw_videos", ["train.mp4"])
    dev_annotation = root / "eval" / "how2sign_realigned_val.csv"
    dev_annotation.parent.mkdir(parents=True)
    with dev_annotation.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("SENTENCE_NAME", "SENTENCE"), delimiter="\t")
        writer.writeheader()
        writer.writerow({"SENTENCE_NAME": "dev", "SENTENCE": "dev text"})
    _touch_videos(root / "eval" / "raw_videos", ["dev.mp4"])
    _touch_videos(root / "test" / "raw_videos", [])
    auxiliary = tmp_path / "labels"
    auxiliary.mkdir()
    with gzip.open(auxiliary / "labels.test", "wb") as handle:
        pickle.dump(
            {"test.mp4": {"video_path": "test.mp4", "text": "test text"}},
            handle,
        )

    result = audit_how2sign(root, auxiliary)
    assert result["status"] == "blocked_assets"
    assert result["splits"]["train"]["ordered_pair_id_hash"]
    assert result["splits"]["test"]["missing_video_count"] == 1
    assert result["test_content_used_for_tuning"] is False


def test_csl_daily_audit_uses_union_inventory_and_disjoint_splits(tmp_path: Path):
    root = tmp_path / "csl"
    names = {"train": "train.mp4", "dev": "dev.mp4", "test": "test.mp4"}
    for split, video in names.items():
        path = root / f"{split}_data_with_num_frames.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("name", "text", "video_path"))
            writer.writeheader()
            writer.writerow({"name": split, "text": f"{split} text", "video_path": video})
    _touch_videos(root / "videos", list(names.values()))

    result = audit_csl_daily(root)
    assert result["status"] == "ready_for_feature_extraction"
    assert result["union_video_inventory"]["referenced_video_count"] == 3
    assert all(value["count"] == 0 for value in result["cross_split_pair_overlap"].values())
