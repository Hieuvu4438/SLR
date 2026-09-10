from __future__ import annotations

import json

from ocem.data.datasets.how2sign import prepare_split
from ocem.data.video import VideoMetadata


def _annotation(path) -> None:
    path.write_text(
        "VIDEO_ID\tVIDEO_NAME\tSENTENCE_ID\tSENTENCE_NAME\tSTART_REALIGNED\tEND_REALIGNED\tSENTENCE\n"
        "source-a\tsource-a-5-rgb_front\tsentence-a\tsentence-a-5-rgb_front\t1.0\t3.0\tKeep This Caption.\n"
        "source-b\tsource-b-5-rgb_front\tsentence-b\tsentence-b-5-rgb_front\t4.0\t5.5\tSecond caption\n",
        encoding="utf-8",
    )


def test_how2sign_records_every_annotation_and_explicit_exclusion(tmp_path) -> None:
    annotation = tmp_path / "annotations.tsv"
    _annotation(annotation)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "sentence-a-5-rgb_front.mp4").write_bytes(b"video")

    def fake_probe(path):
        return VideoMetadata(fps_num=30, fps_den=1, frame_count=60, duration_s=2.0)

    report = prepare_split(
        split="train",
        annotation_path=annotation,
        video_dir=videos,
        raw_relprefix="train/raw_videos",
        output_dir=tmp_path / "output",
        annotation_count=2,
        published_cico_count=1,
        workers=1,
        probe=fake_probe,
    )
    originals = [json.loads(line) for line in (tmp_path / "output/original_train.jsonl").read_text().splitlines()]
    exclusions = [json.loads(line) for line in (tmp_path / "output/exclusions_train.jsonl").read_text().splitlines()]
    assert len(originals) == 2
    assert report["usable_count"] == 1
    assert report["protocol_equivalence"] == "UNVERIFIED"
    assert report["status"] == "BLOCKED_RESOURCE"
    assert originals[0]["caption_raw"] == "Keep This Caption."
    assert originals[0]["video_id"] == "source-a"
    assert exclusions == [
        {"reason": "missing_named_video", "sample_id": "sentence-b-5-rgb_front"}
    ]
