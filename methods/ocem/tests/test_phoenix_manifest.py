from __future__ import annotations

import json

from ocem.data.datasets.phoenix import prepare_split
from ocem.data.video import VideoMetadata


def test_prepare_split_joins_by_id_and_preserves_caption(tmp_path) -> None:
    annotation = tmp_path / "train.csv"
    annotation.write_text(
        "name|video|start|end|speaker|orth|translation\n"
        "b|b/1/*.png|-1|-1|Signer02|GLOSS B|zweite zeile\n"
        "a|a/1/*.png|-1|-1|Signer01|GLOSS A|Erste Zeile!\n",
        encoding="utf-8",
    )
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").write_bytes(b"a")
    (videos / "b.mp4").write_bytes(b"bb")

    def fake_probe(path):
        assert path.name in {"a.mp4", "b.mp4"}
        return VideoMetadata(fps_num=25, fps_den=1, frame_count=50, duration_s=2.0)

    report = prepare_split(
        split="train",
        annotation_path=annotation,
        video_dir=videos,
        output_dir=tmp_path / "output",
        expected_count=2,
        workers=1,
        probe=fake_probe,
    )
    records = [json.loads(line) for line in (tmp_path / "output/train.jsonl").read_text().splitlines()]
    assert report["status"] == "PASS"
    assert [record["sample_id"] for record in records] == ["b", "a"]
    assert records[1]["caption_raw"] == "Erste Zeile!"
    assert records[1]["signer_id"] == "Signer01"
    assert records[1]["frame_count"] == 50


def test_prepare_split_records_missing_video_without_silent_loss(tmp_path) -> None:
    annotation = tmp_path / "dev.csv"
    annotation.write_text(
        "name|video|start|end|speaker|orth|translation\n"
        "missing|missing/1/*.png|-1|-1|Signer01|GLOSS|caption\n",
        encoding="utf-8",
    )
    videos = tmp_path / "videos"
    videos.mkdir()

    def fake_probe(path):
        raise AssertionError("probe result is ignored for a missing file")

    report = prepare_split(
        split="validation",
        annotation_path=annotation,
        video_dir=videos,
        output_dir=tmp_path / "output",
        expected_count=1,
        workers=1,
        probe=fake_probe,
    )
    originals = (tmp_path / "output/original_validation.jsonl").read_text().splitlines()
    usable = (tmp_path / "output/validation.jsonl").read_text().splitlines()
    excluded = [json.loads(line) for line in (tmp_path / "output/exclusions_validation.jsonl").read_text().splitlines()]
    assert report["status"] == "BLOCKED_RESOURCE"
    assert len(originals) == 1
    assert usable == []
    assert excluded == [{"reason": "missing_video", "sample_id": "missing"}]

