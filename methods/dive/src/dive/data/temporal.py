from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class TemporalError(ValueError):
    """Temporal provenance or grid geometry is invalid."""


Interval = tuple[float, float]


@dataclass(frozen=True)
class FrameMap:
    sample_id: str
    duration_sec: float
    raw_frame_indices: tuple[int, ...]
    input_step_intervals_sec: tuple[Interval, ...]
    sampling_policy: str

    def validate(self) -> None:
        if not self.sample_id or not self.sampling_policy:
            raise TemporalError("frame map requires sample ID and sampling policy")
        if self.duration_sec <= 0 or not self.raw_frame_indices:
            raise TemporalError("frame map requires positive duration and at least one step")
        if len(self.raw_frame_indices) != len(self.input_step_intervals_sec):
            raise TemporalError("raw indices and step intervals must have equal length")
        for left, right in self.input_step_intervals_sec:
            if left < 0 or right <= left or right > self.duration_sec + 1e-9:
                raise TemporalError("step intervals must be positive and inside the video")
        if any(a >= b for a, b in zip(self.raw_frame_indices, self.raw_frame_indices[1:])):
            raise TemporalError("raw frame indices must be strictly increasing")
        starts = [interval[0] for interval in self.input_step_intervals_sec]
        if any(a >= b for a, b in zip(starts, starts[1:])):
            raise TemporalError("input-step timestamps must be strictly increasing")


@dataclass(frozen=True)
class CompactFrameMap:
    """Compact identity map for frame-aligned video and pose streams."""

    schema_version: str
    frame_map_key: str
    sample_id: str
    video_frame_count: int
    pose_input_step_count: int
    fps: float
    duration_sec: float
    raw_frame_start: int
    raw_frame_stride: int
    mapping_policy: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CompactFrameMap":
        try:
            record = cls(**dict(raw))
        except TypeError as exc:
            raise TemporalError("invalid compact frame-map fields") from exc
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != "compact_frame_map.v1":
            raise TemporalError("unsupported compact frame-map schema")
        if not self.frame_map_key or not self.sample_id:
            raise TemporalError("compact frame map requires keys and sample ID")
        if self.mapping_policy != "identity_pose_video_frames_v1":
            raise TemporalError("unsupported compact frame-map policy")
        if (
            self.video_frame_count <= 0
            or self.pose_input_step_count != self.video_frame_count
            or self.fps <= 0
            or self.duration_sec <= 0
            or self.raw_frame_start != 0
            or self.raw_frame_stride != 1
        ):
            raise TemporalError("compact frame-map geometry is invalid")
        expected_duration = self.video_frame_count / self.fps
        if abs(self.duration_sec - expected_duration) > 1e-6:
            raise TemporalError("compact frame-map duration differs from frame_count/fps")

    def expand(self) -> FrameMap:
        self.validate()
        intervals = tuple(
            (index / self.fps, (index + 1) / self.fps)
            for index in range(self.pose_input_step_count)
        )
        result = FrameMap(
            sample_id=self.sample_id,
            duration_sec=self.duration_sec,
            raw_frame_indices=tuple(
                self.raw_frame_start + index * self.raw_frame_stride
                for index in range(self.pose_input_step_count)
            ),
            input_step_intervals_sec=intervals,
            sampling_policy=self.mapping_policy,
        )
        result.validate()
        return result


def load_compact_frame_maps(
    path: str | Path, *, expected_sample_ids: Sequence[str] | None = None
) -> tuple[CompactFrameMap, ...]:
    source = Path(path)
    if not source.is_file():
        raise TemporalError(f"compact frame-map artifact does not exist: {source}")
    records: list[CompactFrameMap] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemporalError(f"invalid frame-map JSON at {source}:{line_number}") from exc
        if not isinstance(raw, Mapping):
            raise TemporalError(f"frame-map record at {source}:{line_number} must be an object")
        records.append(CompactFrameMap.from_mapping(raw))
    if not records:
        raise TemporalError(f"compact frame-map artifact is empty: {source}")
    sample_ids = [record.sample_id for record in records]
    keys = [record.frame_map_key for record in records]
    if len(sample_ids) != len(set(sample_ids)) or len(keys) != len(set(keys)):
        raise TemporalError("compact frame maps require unique sample IDs and keys")
    if expected_sample_ids is not None and sample_ids != list(expected_sample_ids):
        raise TemporalError("compact frame-map order/IDs differ from the manifest")
    return tuple(records)


@dataclass(frozen=True)
class ClipGrid:
    sample_id: str
    grid_id: str
    starts: tuple[int, ...]
    valid_steps: tuple[int, ...]
    centers_sec: tuple[float, ...]
    clip_intervals_sec: tuple[Interval, ...]
    clip_steps: int
    stride_steps: int
    max_clips: int
    view_offset_steps: int

    @property
    def num_clips(self) -> int:
        return len(self.starts)


def _select_evenly(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(indices)) != limit:
        raise TemporalError("even selection produced duplicate indices")
    return [values[index] for index in indices]


def build_canonical_grid(
    frame_map: FrameMap,
    *,
    clip_steps: int = 16,
    stride_steps: int = 1,
    max_clips: int = 64,
    view_offset_steps: int = 0,
) -> ClipGrid:
    frame_map.validate()
    if clip_steps <= 0 or stride_steps <= 0 or max_clips <= 0:
        raise TemporalError("grid sizes must be positive")
    count = len(frame_map.raw_frame_indices)
    last_start = max(0, count - clip_steps)
    starts = list(range(0, last_start + 1, stride_steps))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    starts = [min(last_start, max(0, start + view_offset_steps)) for start in starts]
    starts = _select_evenly(list(dict.fromkeys(starts)), max_clips)
    valid_steps = [min(clip_steps, count - start) for start in starts]
    intervals: list[Interval] = []
    centers: list[float] = []
    for start, valid in zip(starts, valid_steps, strict=True):
        left = frame_map.input_step_intervals_sec[start][0]
        right = frame_map.input_step_intervals_sec[start + valid - 1][1]
        intervals.append((left, right))
        centers.append((left + right) / 2)
    payload = {
        "frame_map": frame_map.input_step_intervals_sec,
        "raw_frames": frame_map.raw_frame_indices,
        "sampling_policy": frame_map.sampling_policy,
        "starts": starts,
        "clip_steps": clip_steps,
        "stride_steps": stride_steps,
        "max_clips": max_clips,
        "view_offset_steps": view_offset_steps,
        "alignment_policy": "pose_rgb_same_raw_interval_v1",
    }
    grid_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ClipGrid(
        sample_id=frame_map.sample_id,
        grid_id=grid_id,
        starts=tuple(starts),
        valid_steps=tuple(valid_steps),
        centers_sec=tuple(centers),
        clip_intervals_sec=tuple(intervals),
        clip_steps=clip_steps,
        stride_steps=stride_steps,
        max_clips=max_clips,
        view_offset_steps=view_offset_steps,
    )


def require_distinct_views(first: ClipGrid, second: ClipGrid) -> None:
    if first.sample_id != second.sample_id:
        raise TemporalError("support views must refer to the same sample")
    if first.starts == second.starts and first.clip_intervals_sec == second.clip_intervals_sec:
        raise TemporalError("NO_DISTINCT_SUPPORT_VIEWS")


def pose_receptive_field_intervals(
    frame_map: FrameMap,
    grid: ClipGrid,
    *,
    context_steps_each_side: int = 4,
) -> tuple[Interval, ...]:
    """Map clip RFs back to raw time for a finite local pose dependency."""
    frame_map.validate()
    if grid.sample_id != frame_map.sample_id:
        raise TemporalError("grid/frame map sample IDs differ")
    if context_steps_each_side < 0:
        raise TemporalError("RF context must be nonnegative")
    count = len(frame_map.input_step_intervals_sec)
    result: list[Interval] = []
    for start, valid in zip(grid.starts, grid.valid_steps, strict=True):
        left_index = max(0, start - context_steps_each_side)
        right_index = min(count, start + valid + context_steps_each_side)
        left = frame_map.input_step_intervals_sec[left_index][0]
        right = frame_map.input_step_intervals_sec[right_index - 1][1]
        result.append((left, right))
    return tuple(result)
