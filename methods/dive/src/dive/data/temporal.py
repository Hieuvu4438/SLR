from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


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
