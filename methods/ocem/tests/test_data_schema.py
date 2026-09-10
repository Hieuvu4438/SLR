from __future__ import annotations

import hashlib

import pytest

from ocem.data.schema import SampleRecord, SampleValidationError


def _record(**changes):
    caption = "und nun das wetter"
    values = {
        "schema_version": "ocem.sample.v1",
        "dataset": "phoenix2014t",
        "split": "train",
        "sample_id": "sample-1",
        "video_id": "sample-1",
        "source_group_id": None,
        "signer_id": "Signer01",
        "raw_relpath": "videos/train/sample-1.mp4",
        "caption_raw": caption,
        "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
        "annotation_sha256": hashlib.sha256(b"row").hexdigest(),
        "start_time_s": 0.0,
        "end_time_s": 2.0,
        "fps_num": 25,
        "fps_den": 1,
        "decode_status": "verified_metadata",
        "include_in_protocol": True,
        "exclusion_reason": None,
    }
    values.update(changes)
    return SampleRecord(**values)


def test_valid_sample_round_trip() -> None:
    assert _record().to_dict()["caption_raw"] == "und nun das wetter"


def test_included_sample_requires_valid_interval() -> None:
    with pytest.raises(SampleValidationError, match="positive and increasing"):
        _record(end_time_s=0.0).validate()


def test_excluded_sample_requires_reason() -> None:
    with pytest.raises(SampleValidationError, match="exclusion reason"):
        _record(include_in_protocol=False, decode_status="failed", exclusion_reason=None).validate()

