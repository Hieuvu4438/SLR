from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

from elsc.csl_transfer import build_annotations, translate_dev


def _write_csv(root: Path, split: str, rows: list[dict[str, str]]) -> Path:
    path = root / f"{split}_data_with_num_frames.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "text", "video_path"))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_csl_annotations_preserve_groups_and_do_not_read_test(tmp_path):
    root = tmp_path / "csl"
    _write_csv(
        root,
        "train",
        [
            {"name": "S000001_P0000_T00", "text": "你好", "video_path": "a.mp4"},
            {"name": "S000001_P0004_T00", "text": "你好", "video_path": "b.mp4"},
        ],
    )
    _write_csv(
        root,
        "dev",
        [{"name": "S000002_P0000_T00", "text": "谢谢", "video_path": "c.mp4"}],
    )
    test_marker = root / "test_data_with_num_frames.csv"
    test_marker.write_text("must not be read", encoding="utf-8")

    upstream = tmp_path / "train.pkl"
    with upstream.open("wb") as handle:
        pickle.dump(
            {
                "S000001": [
                    {
                        "video_name": "S000001_P0000_T00",
                        "ori_text": "你好",
                        "text": "Hello",
                    },
                    {
                        "video_name": "S000001_P0004_T00",
                        "ori_text": "你好",
                        "text": "Hello",
                    },
                ]
            },
            handle,
        )
    translations = translate_dev(
        root,
        lambda texts: ["Thank you" for _ in texts],
        model={"id": "translator", "revision": "commit"},
        generation={"num_beams": 4},
    )
    assert translations["test_annotation_accessed"] is False
    translation_path = tmp_path / "translations.json"
    translation_path.write_text(json.dumps(translations), encoding="utf-8")

    report = build_annotations(root, upstream, translation_path, tmp_path / "output")
    assert report["test_annotation_accessed"] is False
    assert report["splits"]["train"]["caption_group_count"] == 1
    assert report["splits"]["dev"]["caption_group_count"] == 1
    with (tmp_path / "output" / "train.pkl").open("rb") as handle:
        train = pickle.load(handle)
    assert train["S000001_P0000_T00"]["caption_id"] == "S000001"
    with (tmp_path / "output" / "dev.pkl").open("rb") as handle:
        dev = pickle.load(handle)
    assert dev["S000002_P0000_T00"]["text"] == "Thank you"

