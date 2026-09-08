from __future__ import annotations

import json
import pickle
from pathlib import Path

from dive.adapters.seds import PINNED_SEDS_COMMIT
from dive.config import load_config
from dive.data.manifest import load_manifest
from dive.data.prepare import _AssetProbe, prepare_how2sign_data
from dive.data.relations import load_excluded_negatives
from dive.data.relevance import load_relevance
from dive.data.temporal import load_compact_frame_maps


HERE = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = HERE / "configs" / "fixture.yaml"


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value))


def _touch_assets(root: Path, split: str, stems: list[str]) -> None:
    layout = {
        "train": ("train/raw_videos", "train/train_pose"),
        "dev": ("eval/raw_videos", "eval/eval_pose"),
        "test": ("test/raw_videos", "test/test_pose"),
    }
    video_dir, pose_dir = layout[split]
    for stem in stems:
        video = root / video_dir / f"{stem}.mp4"
        pose = root / pose_dir / f"{stem}.pkl"
        video.parent.mkdir(parents=True, exist_ok=True)
        pose.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        pose.write_bytes(b"pose")


def test_prepare_data_builds_controlled_disjoint_multiview_protocol(tmp_path, monkeypatch):
    upstream = tmp_path / "SEDS"
    train_path = upstream / "data_h2" / "train.pkl"
    test_path = upstream / "data_h2" / "test.pkl"
    train = {
        "source_train_0": [
            {"text": "same caption", "new_video_name": "train_view_a", "num_frames": 4},
            {"text": "same caption", "new_video_name": "train_view_b", "num_frames": 4},
        ],
        "source_train_1": [
            {"text": "same caption", "new_video_name": "train_duplicate", "num_frames": 4}
        ],
    }
    test = {
        "source_test_0": [
            {"text": "test caption", "new_video_name": "test_view_a", "num_frames": 4},
            {"text": "test caption", "new_video_name": "test_view_b", "num_frames": 4},
        ]
    }
    _write_pickle(train_path, train)
    _write_pickle(test_path, test)
    timing_path = tmp_path / "labels.train"
    timing_rows = [
        {
            "sentence_name": stem,
            "video_id": sentence,
            "start_time": 0.0,
            "end_time": 1.0,
        }
        for sentence, stems in {
            "source_train": ["train_view_a", "train_view_b", "train_duplicate"]
        }.items()
        for stem in stems
    ]
    timing_path.write_text("".join(json.dumps(row) + "\n" for row in timing_rows), encoding="utf-8")
    dev_path = tmp_path / "labels.dev.json"
    dev_path.write_text(
        json.dumps(
            {
                "source_dev_0": {
                    "sentence_name": "dev_view",
                    "text": "Clean &amp; Clear",
                    "video_id": "source_dev",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    assets = tmp_path / "How2Sign"
    _touch_assets(assets, "train", ["train_view_a", "train_view_b", "train_duplicate"])
    _touch_assets(assets, "dev", ["dev_view"])
    _touch_assets(assets, "test", ["test_view_a", "test_view_b"])

    config = load_config(FIXTURE_CONFIG)
    config["data"].update(
        {
            "dataset": "how2sign",
            "upstream_root": str(upstream),
            "train_annotation": str(train_path),
            "train_timing_annotation": str(timing_path),
            "dev_annotation": str(dev_path),
            "test_annotation": str(test_path),
            "video_root": str(assets),
            "pose_root": str(assets),
        }
    )
    config["run"]["output_root"] = str(tmp_path / "runs")
    monkeypatch.setattr(
        "dive.data.prepare._validate_seds_checkout", lambda _root: PINNED_SEDS_COMMIT
    )
    monkeypatch.setattr(
        "dive.data.prepare._probe_asset", lambda _item, _root: _AssetProbe(4, 4, 2.0)
    )

    report = prepare_how2sign_data(config, workers=2)
    assert report["ready"] is True
    assert report["splits"]["train"]["sample_count"] == 3
    assert report["splits"]["train"]["text_count"] == 2
    assert report["splits"]["dev"]["sample_count"] == 1
    assert report["splits"]["test"]["sample_count"] == 2

    shared = tmp_path / "runs/shared/seed17/data"
    train_manifest = load_manifest(shared / "manifests/train.jsonl", expected_split="train")
    assert train_manifest[0].text_id == train_manifest[1].text_id == "source_train_0"
    dev_manifest = load_manifest(shared / "manifests/dev.jsonl", expected_split="dev")
    assert dev_manifest[0].text_original == "Clean &amp; Clear"
    assert dev_manifest[0].text_model == "clean & clear"
    test_manifest = load_manifest(shared / "manifests/test.jsonl", expected_split="test")
    relevance = load_relevance(
        shared / "relevance/test.jsonl",
        video_ids=[record.video_id for record in test_manifest],
        text_ids=[record.text_id for record in test_manifest],
    )
    assert relevance == {
        "test_view_a": frozenset({"source_test_0"}),
        "test_view_b": frozenset({"source_test_0"}),
    }
    frame_maps = load_compact_frame_maps(
        shared / "frame_maps/test.jsonl",
        expected_sample_ids=["test_view_a", "test_view_b"],
    )
    assert all(item.duration_sec == 2.0 for item in frame_maps)
    relations = load_excluded_negatives(
        shared / "relations/train_excluded_negatives.jsonl",
        video_ids=[record.video_id for record in train_manifest],
        text_ids=[record.text_id for record in train_manifest],
        positives_by_video={record.video_id: {record.text_id} for record in train_manifest},
    )
    assert len(relations) == 3
    assert all("exact_normalized_caption_duplicate" in item.reason for item in relations)
    assert (tmp_path / "runs/shared/seed17/run_state.json").is_file()
