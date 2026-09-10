"""Read video stream metadata without decoding or changing source media."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class VideoProbeError(RuntimeError):
    """Raised when a video lacks usable stream metadata."""


@dataclass(frozen=True)
class VideoMetadata:
    fps_num: int
    fps_den: int
    frame_count: int
    duration_s: float


def probe_video(path: str | Path) -> VideoMetadata:
    video_path = Path(path)
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise VideoProbeError(completed.stderr.strip() or "ffprobe failed")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        fps = Fraction(stream["avg_frame_rate"])
        frame_count = int(stream["nb_frames"])
        stream_duration = float(stream["duration"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        raise VideoProbeError(f"invalid ffprobe metadata: {error}") from error
    if fps <= 0 or frame_count <= 0 or stream_duration <= 0:
        raise VideoProbeError("nonpositive fps, frame count, or duration")
    exact_frame_duration = float(Fraction(frame_count * fps.denominator, fps.numerator))
    tolerance = max(1.0 / float(fps), 1e-3)
    if abs(stream_duration - exact_frame_duration) > tolerance:
        raise VideoProbeError(
            f"stream duration {stream_duration} disagrees with frames/fps {exact_frame_duration}"
        )
    return VideoMetadata(
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        frame_count=frame_count,
        duration_s=exact_frame_duration,
    )

