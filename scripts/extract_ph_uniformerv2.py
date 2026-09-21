#!/usr/bin/env python3
"""
Extract UniFormerV2-L/14@336 features for Phoenix14T dataset.
Model: UniFormerV2-L/14@336 (32 frames)
Output feature dimension: 1024D
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add repo root and UniFormerV2 to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "UniFormerV2"))

from third_party.UniFormerV2.extract_features import UniFormerV2FeatureExtractor


FEATURE_DIM = 1024
cv2.setNumThreads(0)


def sliding_window_starts(frame_count: int, clip_frames: int = 32, stride: int = 1) -> list[int]:
    """Calculate sliding window start indices with tail alignment."""
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


def decode_video(path: Path, target_size: int = 336) -> tuple[torch.Tensor, float]:
    """Decode video to RGB tensor normalized with ImageNet stats."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        frames.append(resized)
    cap.release()
    if not frames:
        raise ValueError(f"No frames returned from video: {path}")

    # (T, H, W, C) -> (T, C, H, W)
    arr = np.stack(frames).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, fps


def atomic_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@torch.inference_mode()
def infer_video_features(
    extractor: UniFormerV2FeatureExtractor,
    frames: torch.Tensor,
    starts: list[int],
    clip_len: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Extract features for all sliding windows of a single video."""
    if len(frames) < clip_len:
        padding = frames[-1:].expand(clip_len - len(frames), -1, -1, -1)
        frames = torch.cat([frames, padding], dim=0)

    frames = frames.to(device)
    outputs: list[torch.Tensor] = []
    active_batch = batch_size
    offset = 0

    while offset < len(starts):
        current_starts = starts[offset : offset + active_batch]
        clips = []
        for s in current_starts:
            c = frames[s : s + clip_len]
            if len(c) < clip_len:
                pad = c[-1:].repeat(clip_len - len(c), 1, 1, 1)
                c = torch.cat([c, pad], dim=0)
            clips.append(c)

        batch_tensor = torch.stack(clips, dim=0)  # (B, 32, 3, 336, 336)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                feat = extractor(batch_tensor)  # (B, 1024)
        except torch.cuda.OutOfMemoryError:
            del batch_tensor
            torch.cuda.empty_cache()
            if active_batch == 1:
                raise
            active_batch = max(1, active_batch // 2)
            continue

        outputs.append(feat.float().cpu())
        offset += len(current_starts)

    all_features = torch.cat(outputs, dim=0).numpy()
    return all_features, active_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Extract UniFormerV2 features for Phoenix14T")
    parser.add_argument(
        "--video-root",
        type=str,
        default="/home/dongvk/datasets/phoenix14T/videos_phoenix/videos",
        help="Path to Phoenix14T videos directory",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "features_reextracted" / "ph_uniformerv2"),
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--temporal-root",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "features_reextracted" / "ph_uniformerv2_temporal_metadata"),
        help="Output directory for temporal metadata",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "dev", "train"],
        help="Splits to process in order (default: test dev train)",
    )
    parser.add_argument("--clip-frames", type=int, default=32, help="Number of frames per clip")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    temporal_root = Path(args.temporal_root)
    device = torch.device(args.device)

    print(f"=== UniFormerV2 Feature Extraction ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Video root: {video_root}")
    print(f"Output root: {output_root}")
    print(f"Splits: {args.splits}")
    print(f"Clip frames: {args.clip_frames}, Stride: {args.stride}, Batch size: {args.batch_size}")

    print("\nLoading UniFormerV2 model...")
    extractor = UniFormerV2FeatureExtractor(device=args.device)
    print("Model loaded successfully!")

    total_start_time = time.time()
    active_batch = args.batch_size

    for split in args.splits:
        split_video_dir = video_root / split
        if not split_video_dir.is_dir():
            print(f"Warning: Split directory does not exist: {split_video_dir}, skipping.")
            continue

        video_files = sorted(list(split_video_dir.glob("*.mp4")))
        total_videos = len(video_files)
        print(f"\n--- Processing split '{split}': {total_videos} videos ---")

        split_out_dir = output_root / split
        split_temp_dir = temporal_root / split
        split_out_dir.mkdir(parents=True, exist_ok=True)
        split_temp_dir.mkdir(parents=True, exist_ok=True)

        completed = 0
        resumed = 0
        split_start_time = time.time()

        for idx, video_path in enumerate(video_files, 1):
            name = video_path.stem
            feat_path = split_out_dir / f"{name}.pkl"
            meta_path = split_out_dir / f"{name}.pkl.meta.json"
            temp_path = split_temp_dir / f"{name}.json"

            # Check if already extracted
            if feat_path.is_file() and meta_path.is_file():
                resumed += 1
                completed += 1
                continue

            v_start = time.time()
            try:
                frames, fps = decode_video(video_path, target_size=336)
                frame_count = len(frames)
                starts = sliding_window_starts(frame_count, args.clip_frames, args.stride)
                features, active_batch = infer_video_features(
                    extractor, frames, starts, args.clip_frames, device, active_batch
                )

                # Save pickle
                payload = {"name": str(video_path.resolve()), "feature": features}
                atomic_pickle(payload, feat_path)

                # Save meta
                sidecar = {
                    "schema_version": 1,
                    "model": "UniFormerV2-L/14@336",
                    "source_video": str(video_path.resolve()),
                    "feature_shape": list(features.shape),
                    "feature_dtype": str(features.dtype),
                    "effective_batch_size": active_batch,
                }
                atomic_json_dump(sidecar, meta_path)

                # Save temporal metadata
                temporal = {
                    "source_video": str(video_path.resolve()),
                    "decoded_frame_count": frame_count,
                    "fps": fps,
                    "clip_frames": args.clip_frames,
                    "stride": args.stride,
                    "rf_start": starts,
                    "rf_end": [min(s + args.clip_frames, frame_count) for s in starts],
                }
                atomic_json_dump(temporal, temp_path)

                del frames, features
                torch.cuda.empty_cache()

                completed += 1
                v_elapsed = time.time() - v_start
                elapsed_split = time.time() - split_start_time
                remaining_videos = total_videos - completed
                avg_time = elapsed_split / max(1, completed - resumed)
                eta_sec = remaining_videos * avg_time
                eta_hours = eta_sec / 3600

                if completed % 10 == 0 or completed == total_videos or completed <= 5:
                    print(
                        f"[{split} {completed}/{total_videos}] {name} ({frame_count} frames, {len(starts)} clips) "
                        f"- {v_elapsed:.1f}s | Avg: {avg_time:.1f}s/vid | ETA: {eta_hours:.1f}h"
                    )

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue

        print(f"Split '{split}' completed! Processed: {completed}/{total_videos} (Resumed: {resumed}).")

    total_hours = (time.time() - total_start_time) / 3600
    print(f"\nAll splits completed in {total_hours:.2f} hours!")


if __name__ == "__main__":
    main()
