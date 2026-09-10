"""Canonical sample-manifest schema."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


class SampleValidationError(ValueError):
    """Raised when a sample cannot satisfy the manifest contract."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SampleRecord:
    schema_version: str
    dataset: str
    split: str
    sample_id: str
    video_id: str
    source_group_id: str | None
    signer_id: str | None
    raw_relpath: str
    caption_raw: str
    caption_sha256: str
    annotation_sha256: str
    start_time_s: float
    end_time_s: float
    fps_num: int
    fps_den: int
    decode_status: str
    include_in_protocol: bool
    exclusion_reason: str | None
    frame_count: int | None = None
    raw_bytes: int | None = None
    source_annotation: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.schema_version != "ocem.sample.v1":
            raise SampleValidationError("unsupported sample schema")
        for field_name in ("dataset", "split", "sample_id", "video_id", "raw_relpath"):
            if not getattr(self, field_name):
                raise SampleValidationError(f"{field_name} must be non-empty")
        if not _SHA256.fullmatch(self.caption_sha256):
            raise SampleValidationError("caption_sha256 must be lowercase SHA-256")
        if not _SHA256.fullmatch(self.annotation_sha256):
            raise SampleValidationError("annotation_sha256 must be lowercase SHA-256")
        if self.include_in_protocol:
            if self.decode_status != "verified_metadata":
                raise SampleValidationError("included sample must have verified video metadata")
            if not all(math.isfinite(value) for value in (self.start_time_s, self.end_time_s)):
                raise SampleValidationError("included sample interval must be finite")
            if self.start_time_s < 0 or self.end_time_s <= self.start_time_s:
                raise SampleValidationError("included sample interval must be positive and increasing")
            if self.fps_num <= 0 or self.fps_den <= 0:
                raise SampleValidationError("included sample FPS must be a positive rational")
            if self.exclusion_reason is not None:
                raise SampleValidationError("included sample cannot have an exclusion reason")
        elif not self.exclusion_reason:
            raise SampleValidationError("excluded sample must have an exclusion reason")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

