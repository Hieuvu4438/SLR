from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from slr_common.resources import require_resources, require_storage_budget
from slr_common.utils import atomic_json_dump, sha256_file, sha256_json


FEATURE_DIM = 1024

# Match the pinned upstream extractor and avoid each concurrent extraction
# process creating an OpenCV worker pool over all host cores.
cv2.setNumThreads(0)


@dataclass(frozen=True)
class ExtractionRecipe:
    schema_version: int = 1
    clip_frames: int = 16
    stride: int = 1
    resize_short_side: int = 256
    crop_size: int = 224
    color_order: str = "rgb"
    input_scale: str = "uint8_to_float32_div_255"
    normalization_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalization_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    spatial_mode: str = "square_lanczos_then_cico_gpu_center_resample"
    temporal_mode: str = "upstream_cico_sliding_window"
    output_dtype: str = "float32"

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported extraction recipe schema")
        if self.clip_frames < 1 or self.stride < 1:
            raise ValueError("clip_frames and stride must be positive integers")
        if self.resize_short_side < self.crop_size or self.crop_size < 1:
            raise ValueError("resize_short_side must be at least crop_size")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


def sliding_window_starts(frame_count: int, clip_frames: int = 16, stride: int = 1) -> list[int]:
    """Match CiCo's PH feature-extractor window placement exactly."""

    if frame_count < 1 or clip_frames < 1 or stride < 1:
        raise ValueError("frame_count, clip_frames, and stride must be positive")
    num_clips = math.ceil((frame_count - clip_frames) / stride) + 1
    num_clips = max(num_clips, 1)
    starts: list[int] = []
    for index in range(num_clips):
        proposed = index * stride
        actual_length = min(clip_frames, frame_count - proposed)
        if actual_length == clip_frames:
            start = proposed
        elif frame_count - clip_frames >= 0:
            start = frame_count - clip_frames
        else:
            start = 0
        starts.append(int(start))
    return starts


def preprocess_rgb_frame(
    frame: np.ndarray, *, resize_short_side: int = 256, crop_size: int = 224
) -> np.ndarray:
    """Prepare one frame for CiCo's default 256-pixel GPU collation path.

    The public PH loader defaults to ``gpu_collation=256`` and its collater then
    applies the 224-pixel center resampling on the GPU.  The released loader does
    not define how a non-square PH frame reaches that 256x256 buffer; Lanczos
    square resizing is the closest reproducible match to the released agnostic
    features and is therefore pinned explicitly here.
    """

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {frame.shape}")
    if resize_short_side < crop_size:
        raise ValueError("resize_short_side must be at least crop_size")
    rgb = frame.astype(np.float32, copy=False)
    if np.issubdtype(frame.dtype, np.integer):
        rgb = rgb / 255.0
    if not np.isfinite(rgb).all() or float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise ValueError("RGB values must be finite and in [0,1]")
    resized = cv2.resize(
        rgb,
        (resize_short_side, resize_short_side),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count < 1:
        raise ValueError(f"video has no frames: {path}")
    return count


def decode_video(path: Path, recipe: ExtractionRecipe) -> tuple[torch.Tensor, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = bgr[:, :, ::-1]
        frames.append(
            preprocess_rgb_frame(
                rgb,
                resize_short_side=recipe.resize_short_side,
                crop_size=recipe.crop_size,
            )
        )
    capture.release()
    if not frames:
        raise ValueError(f"video decoder returned no frames: {path}")
    return torch.from_numpy(np.stack(frames)), fps


def _import_i3d(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"upstream I3D implementation is missing: {path}")
    spec = importlib.util.spec_from_file_location("elsc_pinned_upstream_i3d", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import upstream I3D implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_i3d(checkpoint_path: Path, implementation_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("I3D checkpoint must contain a state_dict")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint["state_dict"].items()
    }
    classifier = state.get("logits.conv3d.weight")
    if classifier is None or classifier.ndim != 5 or classifier.shape[1] != FEATURE_DIM:
        raise ValueError("cannot infer a compatible I3D classifier from checkpoint")
    module = _import_i3d(implementation_path)
    model = module.InceptionI3d(
        num_classes=int(classifier.shape[0]),
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=16,
        include_embds=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def _window_batches(starts: list[int], batch_size: int) -> Iterator[list[int]]:
    for offset in range(0, len(starts), batch_size):
        yield starts[offset : offset + batch_size]


@torch.inference_mode()
def infer_video_features(
    model,
    frames: torch.Tensor,
    starts: list[int],
    recipe: ExtractionRecipe,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    if frames.ndim != 4 or frames.shape[1:] != (
        3,
        recipe.resize_short_side,
        recipe.resize_short_side,
    ):
        raise ValueError(f"unexpected preprocessed video shape: {tuple(frames.shape)}")
    if len(frames) < recipe.clip_frames:
        padding = frames[-1:].expand(recipe.clip_frames - len(frames), -1, -1, -1)
        frames = torch.cat((frames, padding), dim=0)
    # Transfer each decoded video only once. Constructing highly overlapping
    # windows on the host would repeatedly copy the same 256x256 frames into
    # every batch and leave the GPU starved on long PH clips.
    frames = frames.to(device)
    outputs: list[torch.Tensor] = []
    active_batch = batch_size
    offset = 0
    while offset < len(starts):
        current = starts[offset : offset + active_batch]
        indices = torch.as_tensor(current, dtype=torch.long, device=device)[:, None]
        indices = indices + torch.arange(recipe.clip_frames, device=device)[None, :]
        clips = frames.index_select(0, indices.flatten()).reshape(
            len(current),
            recipe.clip_frames,
            3,
            recipe.resize_short_side,
            recipe.resize_short_side,
        )
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        crop_scale = recipe.crop_size / recipe.resize_short_side
        ticks = torch.linspace(
            -crop_scale,
            crop_scale,
            steps=recipe.crop_size,
            device=device,
            dtype=clips.dtype,
        )
        grid_y, grid_x = torch.meshgrid(ticks, ticks, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)
        clips = F.grid_sample(
            clips.reshape(len(current), 3 * recipe.clip_frames, recipe.resize_short_side, recipe.resize_short_side),
            grid=grid.expand(len(current), -1, -1, -1),
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        ).reshape(
            len(current), 3, recipe.clip_frames, recipe.crop_size, recipe.crop_size
        )
        clips.sub_(0.5)
        try:
            encoded = model(clips)["embds"].flatten(1)
        except torch.cuda.OutOfMemoryError:
            del clips
            torch.cuda.empty_cache()
            if active_batch == 1:
                raise
            active_batch = max(1, active_batch // 2)
            continue
        if encoded.shape != (len(current), FEATURE_DIM):
            raise ValueError(f"unexpected I3D embedding shape: {tuple(encoded.shape)}")
        outputs.append(encoded.float().cpu())
        offset += len(current)
    return torch.cat(outputs).numpy(), active_batch


def _atomic_pickle(value: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _temporal_metadata(
    source: Path,
    source_sha256: str,
    frame_count: int,
    fps: float,
    starts: list[int],
    recipe: ExtractionRecipe,
) -> dict[str, Any]:
    # Keep the in-memory value identical to its JSON representation so a second
    # feature stream can safely verify and reuse the shared temporal sidecar.
    recipe_json = json.loads(json.dumps(asdict(recipe)))
    return {
        "schema_version": 1,
        "verified": True,
        "verification_scope": "generated_by_deterministic_elsc_i3d_extractor",
        "coordinate_system": "input_frame",
        "interval_convention": "half_open",
        "source_video": str(source.resolve()),
        "source_video_sha256": source_sha256,
        "decoded_frame_count": frame_count,
        "fps": fps,
        "recipe": recipe_json,
        "recipe_sha256": recipe.digest,
        "rf_start": starts,
        "rf_end": [min(start + recipe.clip_frames, frame_count) for start in starts],
    }


def _is_complete(
    feature_path: Path,
    sidecar_path: Path,
    temporal_path: Path,
    *,
    source: Path,
    source_sha256: str,
    checkpoint_sha256: str,
    recipe: ExtractionRecipe,
    expected_windows: int,
) -> bool:
    if not feature_path.is_file() or not sidecar_path.is_file() or not temporal_path.is_file():
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
        expected = {
            "source_video": str(source.resolve()),
            "source_video_sha256": source_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "recipe_sha256": recipe.digest,
            "feature_shape": [expected_windows, FEATURE_DIM],
        }
        if any(sidecar.get(key) != value for key, value in expected.items()):
            return False
        if (
            temporal.get("verified") is not True
            or temporal.get("source_video_sha256") != source_sha256
            or temporal.get("recipe_sha256") != recipe.digest
            or len(temporal.get("rf_start", [])) != expected_windows
            or len(temporal.get("rf_end", [])) != expected_windows
        ):
            return False
        return sidecar.get("feature_sha256") == sha256_file(feature_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _parse_split_video_lists(
    assignments: list[str], splits: list[str]
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for assignment in assignments:
        split, separator, raw_path = assignment.partition("=")
        if not separator or split not in {"train", "dev", "test"} or not raw_path:
            raise ValueError(
                "--split-video-list must use SPLIT=PATH with split train, dev, or test"
            )
        if split in result:
            raise ValueError(f"duplicate video-list assignment for split={split}")
        result[split] = Path(raw_path)
    if result and set(result) != set(splits):
        raise ValueError(
            "video-list assignments must exactly match requested splits: "
            f"assigned={sorted(result)}, requested={sorted(splits)}"
        )
    return result


def _listed_videos(video_root: Path, split: str, list_path: Path) -> list[Path]:
    if not list_path.is_file():
        raise FileNotFoundError(f"video list is missing for split={split}: {list_path}")
    videos: list[Path] = []
    seen: set[str] = set()
    with list_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            name = line.rstrip("\n\r")
            candidate = Path(name)
            if (
                not name
                or candidate.is_absolute()
                or len(candidate.parts) != 1
                or candidate.suffix.lower() != ".mp4"
            ):
                raise ValueError(
                    f"invalid relative MP4 filename at {list_path}:{line_number}: {name!r}"
                )
            if name in seen:
                raise ValueError(f"duplicate filename at {list_path}:{line_number}: {name}")
            source = video_root / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"listed video is missing for split={split}: {source}"
                )
            seen.add(name)
            videos.append(source)
    if not videos:
        raise ValueError(f"video list is empty for split={split}: {list_path}")
    return videos


def _collect_plan(
    video_root: Path,
    splits: list[str],
    recipe: ExtractionRecipe,
    split_video_lists: dict[str, Path] | None = None,
):
    plan: list[tuple[str, Path, int, int]] = []
    seen_sources: dict[str, str] = {}
    for split in splits:
        if split_video_lists:
            videos = _listed_videos(video_root, split, split_video_lists[split])
        else:
            split_root = video_root / split
            videos = sorted(split_root.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"no MP4 videos found for split={split}")
        for video in videos:
            source_key = str(video.resolve())
            previous_split = seen_sources.setdefault(source_key, split)
            if previous_split != split:
                raise ValueError(
                    f"video appears in multiple splits: {video} ({previous_split}, {split})"
                )
            frames = _video_frame_count(video)
            windows = len(sliding_window_starts(frames, recipe.clip_frames, recipe.stride))
            plan.append((split, video, frames, windows))
    return plan


def _acquire_extraction_lock(
    lock: Any,
    lock_path: Path,
    *,
    wait_seconds: float,
    poll_seconds: float,
) -> float:
    """Acquire a split lock, waiting for an intentional peer extractor if needed."""
    started = time.monotonic()
    deadline = started + wait_seconds
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return time.monotonic() - started
        except BlockingIOError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"timed out after {wait_seconds:.0f}s waiting for extractor lock "
                    f"{lock_path}"
                ) from error
            sleep_seconds = min(poll_seconds, remaining)
            print(
                f"waiting_for_extractor_lock path={lock_path} "
                f"remaining_seconds={remaining:.0f}",
                flush=True,
            )
            time.sleep(sleep_seconds)


def extract(args: argparse.Namespace) -> dict[str, Any]:
    recipe = ExtractionRecipe(stride=args.stride)
    recipe.validate()
    video_root = Path(args.video_root)
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    temporal_root = Path(args.temporal_metadata_root)
    implementation = Path(args.upstream_i3d)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha256} != "
            f"{args.expected_checkpoint_sha256}"
        )
    split_video_lists = _parse_split_video_lists(args.split_video_list, args.splits)
    plan = _collect_plan(video_root, args.splits, recipe, split_video_lists)
    raw_feature_bytes = sum(item[3] for item in plan) * FEATURE_DIM * 4
    missing_feature_bytes = sum(
        windows * FEATURE_DIM * 4
        for split, video, _frames, windows in plan
        if not (output_root / split / f"{video.stem}.pkl").is_file()
    )
    largest_feature_bytes = max(item[3] for item in plan) * FEATURE_DIM * 4
    # Include JSON/pickle overhead plus one full atomic replacement temporary.
    planned_write_bytes = math.ceil(missing_feature_bytes * 1.05) + largest_feature_bytes
    storage = require_storage_budget(
        output_root,
        planned_write_bytes=planned_write_bytes,
        min_remaining_gib=args.min_free_disk_gib,
        operation=f"{args.stream_name} I3D extraction",
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if args.dry_run else "running",
        "stream_name": args.stream_name,
        "video_root": str(video_root.resolve()),
        "output_root": str(output_root.resolve()),
        "temporal_metadata_root": str(temporal_root.resolve()),
        "splits": args.splits,
        "split_video_lists": {
            split: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for split, path in split_video_lists.items()
        },
        "videos": len(plan),
        "windows": sum(item[3] for item in plan),
        "raw_feature_bytes": raw_feature_bytes,
        "missing_raw_feature_bytes": missing_feature_bytes,
        "planned_write_bytes_with_atomic_margin": planned_write_bytes,
        "storage": storage,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "upstream_i3d": str(implementation.resolve()),
        "upstream_i3d_sha256": sha256_file(implementation),
        "recipe": asdict(recipe),
        "recipe_sha256": recipe.digest,
        "completed": 0,
        "resumed": 0,
        "failed": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    split_label = "_".join(args.splits)
    report_path = output_root / f"extraction_report_{split_label}.json"
    if args.dry_run:
        atomic_json_dump(summary, report_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("I3D extraction requires an available CUDA device")
    # Different dataset splits write to disjoint directories and reports, so
    # they may safely use the large GPU concurrently. Keep exclusion within a
    # split to prevent duplicate work and report/sidecar races for the same videos.
    lock_path = output_root / f".extract-{split_label}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        summary["lock_wait_seconds"] = _acquire_extraction_lock(
            lock,
            lock_path,
            wait_seconds=args.lock_wait_seconds,
            poll_seconds=args.lock_poll_seconds,
        )
        summary["resources_at_start"] = require_resources(
            output_root,
            device,
            min_disk_gib=args.min_free_disk_gib,
            min_gpu_gib=args.min_free_gpu_gib,
            operation=f"{args.stream_name} I3D extraction",
        )
        atomic_json_dump(summary, report_path)
        model = load_i3d(checkpoint_path, implementation, device)
        active_batch = args.batch_size
        started = time.monotonic()
        for index, (split, source, expected_frames, expected_windows) in enumerate(plan, 1):
            name = source.stem
            feature_path = output_root / split / f"{name}.pkl"
            sidecar_path = output_root / split / f"{name}.pkl.meta.json"
            temporal_path = temporal_root / split / f"{name}.json"
            source_sha256 = sha256_file(source)
            if _is_complete(
                feature_path,
                sidecar_path,
                temporal_path,
                source=source,
                source_sha256=source_sha256,
                checkpoint_sha256=checkpoint_sha256,
                recipe=recipe,
                expected_windows=expected_windows,
            ):
                summary["resumed"] += 1
                summary["completed"] += 1
                continue
            frames: torch.Tensor | None = None
            try:
                frames, fps = decode_video(source, recipe)
                frame_count = len(frames)
                if frame_count != expected_frames:
                    expected_frames = frame_count
                starts = sliding_window_starts(frame_count, recipe.clip_frames, recipe.stride)
                features, active_batch = infer_video_features(
                    model, frames, starts, recipe, device, active_batch
                )
                payload = {"name": str(source.resolve()), "feature": features}
                _atomic_pickle(payload, feature_path)
                feature_sha256 = sha256_file(feature_path)
                sidecar = {
                    "schema_version": 1,
                    "stream_name": args.stream_name,
                    "source_video": str(source.resolve()),
                    "source_video_sha256": source_sha256,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "recipe_sha256": recipe.digest,
                    "feature_shape": list(features.shape),
                    "feature_dtype": str(features.dtype),
                    "feature_sha256": feature_sha256,
                    "effective_batch_size": active_batch,
                }
                atomic_json_dump(sidecar, sidecar_path)
                temporal = _temporal_metadata(
                    source, source_sha256, frame_count, fps, starts, recipe
                )
                if temporal_path.is_file():
                    existing = json.loads(temporal_path.read_text(encoding="utf-8"))
                    if existing != temporal:
                        raise ValueError(f"conflicting temporal metadata: {temporal_path}")
                else:
                    atomic_json_dump(temporal, temporal_path)
                summary["completed"] += 1
            except Exception as error:
                summary["failed"].append({"video": str(source), "error": repr(error)})
                summary["status"] = "failed"
                atomic_json_dump(summary, report_path)
                raise
            finally:
                if frames is not None:
                    del frames
            if index % args.report_interval == 0:
                summary["elapsed_seconds"] = time.monotonic() - started
                summary["effective_batch_size"] = active_batch
                atomic_json_dump(summary, report_path)
                print(
                    f"[{args.stream_name}] {index}/{len(plan)} videos; "
                    f"completed={summary['completed']} resumed={summary['resumed']}",
                    flush=True,
                )
        summary["status"] = "complete"
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["effective_batch_size"] = active_batch
        atomic_json_dump(summary, report_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deterministic dense CiCo-compatible I3D features from local videos"
    )
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--stream-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--temporal-metadata-root", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), required=True)
    parser.add_argument(
        "--split-video-list",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help="read exact flat-root MP4 filenames for each requested split",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-free-disk-gib", type=float, default=20.0)
    parser.add_argument("--min-free-gpu-gib", type=float, default=12.0)
    parser.add_argument("--report-interval", type=int, default=25)
    parser.add_argument("--lock-wait-seconds", type=float, default=86_400.0)
    parser.add_argument("--lock-poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--upstream-i3d",
        default=(
            "third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.report_interval < 1:
        raise ValueError("batch-size and report-interval must be positive")
    if args.lock_wait_seconds < 0 or args.lock_poll_seconds <= 0:
        raise ValueError("lock-wait-seconds must be nonnegative and lock-poll-seconds positive")
    extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
