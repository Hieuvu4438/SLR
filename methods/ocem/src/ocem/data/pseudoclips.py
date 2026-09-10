"""Manifest-indexed raw-frame loader for P14T pseudo-label adaptation clips."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class PseudoClipError(ValueError):
    """Raised when a pseudo-clip index or raw-frame decode violates its contract."""


def load_pseudoclip_index(
    path: str | Path,
    *,
    expected_sha256: str,
    adaptation_split: str,
) -> list[dict[str, Any]]:
    """Load and validate the locked index, returning exactly one adaptation split."""

    if adaptation_split not in {"train", "holdout"}:
        raise PseudoClipError("adaptation_split must be train or holdout")
    path = Path(path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PseudoClipError(f"pseudo-label index SHA-256 mismatch: {actual_sha256}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, int, int]] = set()
    all_splits: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PseudoClipError(f"cannot read pseudo-label index {path}: {error}") from error
    if not lines:
        raise PseudoClipError(f"pseudo-label index is empty: {path}")
    for line_number, line in enumerate(lines, start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise PseudoClipError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(item, Mapping):
            raise PseudoClipError(f"pseudo-label record is not an object at line {line_number}")
        if item.get("schema_version") != "ocem.p14t_pseudolabel.v1":
            raise PseudoClipError(f"unsupported pseudo-label schema at line {line_number}")
        pseudo_id = str(item.get("pseudo_id", ""))
        sample_id = str(item.get("source_sample_id", ""))
        split = str(item.get("adaptation_split", ""))
        source_video = Path(str(item.get("source_video", "")))
        try:
            class_index = int(item["class_index"])
            anchor = int(item["anchor_start_frame"])
            support_start = int(item["support_start_frame"])
            support_end = int(item["support_end_frame_exclusive"])
            clip_start = int(item["upstream_clip_start_frame"])
            clip_end = int(item["upstream_clip_end_frame_exclusive"])
            confidence = float(item["selection_confidence"])
        except (KeyError, TypeError, ValueError) as error:
            raise PseudoClipError(f"malformed pseudo-label fields at line {line_number}") from error
        if not pseudo_id or pseudo_id in seen_ids:
            raise PseudoClipError(f"missing or duplicate pseudo_id at line {line_number}")
        if not sample_id or split not in {"train", "holdout"}:
            raise PseudoClipError(f"missing sample ID or invalid split at line {line_number}")
        if not source_video.is_file():
            raise PseudoClipError(f"source video is missing at line {line_number}: {source_video}")
        if source_video.stem != sample_id:
            raise PseudoClipError(f"source video/sample ID mismatch at line {line_number}")
        if class_index < 0 or class_index >= 5383:
            raise PseudoClipError(f"class index out of range at line {line_number}")
        if not math.isfinite(confidence) or confidence <= 0.6 or confidence > 1.0:
            raise PseudoClipError(f"invalid selection confidence at line {line_number}")
        if (
            support_start < 0
            or support_end <= support_start
            or clip_start != support_start
            or clip_end != support_end - 1
            or clip_end <= clip_start
            or anchor < support_start
            or anchor >= support_end
        ):
            raise PseudoClipError(f"invalid support/materialization interval at line {line_number}")
        identity = (sample_id, class_index, anchor)
        if identity in seen_identities:
            raise PseudoClipError(f"duplicate source/class/anchor identity at line {line_number}")
        expected_id = hashlib.sha256(
            (
                "ocem-p14t-pseudo-v1\0"
                + canonical_json_sha256(
                    {
                        "source_sample_id": sample_id,
                        "class_index": class_index,
                        "anchor_start": anchor,
                    }
                )
            ).encode()
        ).hexdigest()
        if pseudo_id != expected_id:
            raise PseudoClipError(f"pseudo_id content digest mismatch at line {line_number}")
        seen_ids.add(pseudo_id)
        seen_identities.add(identity)
        all_splits.add(split)
        if split == adaptation_split:
            records.append(dict(item))
    if all_splits != {"train", "holdout"}:
        raise PseudoClipError(f"full pseudo-label index lacks a required split: {all_splits}")
    if not records:
        raise PseudoClipError(f"pseudo-label index has no {adaptation_split} records")
    return records


def temporal_sample_start(
    record: Mapping[str, Any],
    *,
    training: bool,
    rng: random.Random | None = None,
    clip_frames: int = 16,
) -> int:
    """Choose a source-frame start with CiCo's random-train/center-val policy."""

    start = int(record["upstream_clip_start_frame"])
    end = int(record["upstream_clip_end_frame_exclusive"])
    duration = end - start
    if duration <= 0 or clip_frames < 1:
        raise PseudoClipError("pseudo clip and clip_frames must have positive duration")
    maximum_offset = max(0, duration - clip_frames)
    if training:
        generator = rng if rng is not None else random
        return start + generator.randint(0, maximum_offset)
    return start + maximum_offset // 2


def _decode_bgr_frames(
    path: Path,
    *,
    start: int,
    count: int,
    sequential_from_zero: bool,
) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as error:
        raise PseudoClipError("OpenCV is required for pseudo-clip decoding") from error
    cv2.setNumThreads(0)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PseudoClipError(f"cannot open source video: {path}")
    if not sequential_from_zero:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    position = 0 if sequential_from_zero else start
    target_end = start + count
    while position < target_end:
        ok, frame = capture.read()
        if not ok:
            break
        if position >= start:
            frames.append(frame)
        position += 1
    capture.release()
    if len(frames) != count:
        raise PseudoClipError(
            f"decoded {len(frames)}/{count} requested frames from {path} at start {start}"
        )
    return frames


def decode_pseudoclip(
    record: Mapping[str, Any],
    *,
    training: bool,
    rng: random.Random | None = None,
    clip_frames: int = 16,
    resize: int = 224,
    sequential_from_zero: bool = False,
):
    """Decode raw source frames using the pseudo-MP4's frame-selection semantics.

    CiCo first writes ``frames[start:max_frame]`` to an ``mp4v`` pseudo-video,
    then samples/pads that file. This loader selects the same source frame indices
    but deliberately removes the lossy encode/decode round trip.
    """

    try:
        import torch
        from PIL import Image
    except ImportError as error:
        raise PseudoClipError("PyTorch and Pillow are required for pseudo-clip decoding") from error
    selected_start = temporal_sample_start(
        record, training=training, rng=rng, clip_frames=clip_frames
    )
    materialized_end = int(record["upstream_clip_end_frame_exclusive"])
    observed_count = min(clip_frames, materialized_end - selected_start)
    bgr_frames = _decode_bgr_frames(
        Path(str(record["source_video"])),
        start=selected_start,
        count=observed_count,
        sequential_from_zero=sequential_from_zero,
    )
    rgb_frames: list[np.ndarray] = []
    for bgr in bgr_frames:
        resized_bgr = np.asarray(Image.fromarray(bgr).resize((resize, resize)))
        rgb_frames.append(np.ascontiguousarray(resized_bgr[:, :, ::-1]))
    while len(rgb_frames) < clip_frames:
        rgb_frames.append(rgb_frames[-1].copy())
    array = np.stack(rgb_frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    return {
        "rgb": tensor,
        "class": int(record["class_index"]),
        "pseudo_id": str(record["pseudo_id"]),
        "source_sample_id": str(record["source_sample_id"]),
        "selected_start_frame": selected_start,
        "observed_frame_count": observed_count,
        "padded_frame_count": clip_frames - observed_count,
        "codec_roundtrip": False,
    }


class PseudoClipDataset:
    """Minimal torch-compatible dataset backed by a validated pseudo-label index."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        training: bool,
        rng: random.Random | None = None,
        seed: int | None = None,
        epoch: int = 0,
    ) -> None:
        if not records:
            raise PseudoClipError("PseudoClipDataset requires at least one record")
        if rng is not None and seed is not None:
            raise PseudoClipError("provide either a stateful rng or deterministic seed, not both")
        if seed is not None and (seed < 0 or epoch < 0):
            raise PseudoClipError("dataset seed and epoch must be nonnegative")
        self.records = [dict(record) for record in records]
        self.training = training
        self.rng = rng
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self.rng
        if self.training and self.seed is not None:
            pseudo_id = str(self.records[index]["pseudo_id"])
            digest = hashlib.sha256(
                f"ocem-pseudoclip-sampling-v1\0{self.seed}\0{self.epoch}\0{pseudo_id}".encode()
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
        return decode_pseudoclip(self.records[index], training=self.training, rng=rng)


def validate_pseudoclip_loader(
    *,
    index: str | Path,
    expected_index_sha256: str,
    adaptation_split: str,
    samples: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Smoke-test real indexed clips and verify random access against sequential decode."""

    if samples < 1 or seed < 0:
        raise PseudoClipError("samples must be positive and seed nonnegative")
    index = Path(index)
    records = load_pseudoclip_index(
        index, expected_sha256=expected_index_sha256, adaptation_split=adaptation_split
    )
    chosen = records[: min(samples, len(records))]
    seek_equal = True
    batch = []
    decode_records = []
    for position, record in enumerate(chosen):
        rng_seed = seed + position
        seek = decode_pseudoclip(
            record, training=adaptation_split == "train", rng=random.Random(rng_seed)
        )
        sequential = decode_pseudoclip(
            record,
            training=adaptation_split == "train",
            rng=random.Random(rng_seed),
            sequential_from_zero=True,
        )
        equal = bool(np.array_equal(seek["rgb"].numpy(), sequential["rgb"].numpy()))
        seek_equal = seek_equal and equal
        batch.append(seek["rgb"])
        decode_records.append(
            {
                "pseudo_id": seek["pseudo_id"],
                "source_sample_id": seek["source_sample_id"],
                "selected_start_frame": seek["selected_start_frame"],
                "observed_frame_count": seek["observed_frame_count"],
                "padded_frame_count": seek["padded_frame_count"],
                "seek_equals_sequential": equal,
            }
        )
    try:
        import torch
    except ImportError as error:
        raise PseudoClipError("PyTorch is required for pseudo-clip validation") from error
    tensor = torch.stack(batch)
    status = "PASS" if seek_equal and bool(torch.isfinite(tensor).all()) else "FAIL_TECHNICAL"
    return {
        "schema_version": "ocem.p14t_pseudoclip_loader_validation.v1",
        "status": status,
        "index": {"path": str(index.resolve()), "sha256": expected_index_sha256},
        "adaptation_split": adaptation_split,
        "indexed_records": len(records),
        "tested_records": len(chosen),
        "batch_shape": list(tensor.shape),
        "batch_dtype": str(tensor.dtype),
        "batch_min": float(tensor.min()),
        "batch_max": float(tensor.max()),
        "batch_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "seek_equals_sequential_for_all": seek_equal,
        "decode_records": decode_records,
        "frame_selection_compatibility": "cico_pseudo_mp4_interval_sampling_and_padding",
        "intentional_difference": "lossy_mp4v_pseudo-video_roundtrip_removed",
        "ready_for_one_batch_update_parity": status == "PASS",
        "ready_for_adaptation_training": False,
        "training_blocker": "One-batch loss/gradient/update parity has not passed yet.",
    }
