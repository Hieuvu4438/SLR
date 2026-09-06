from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from elsc.features.i3d import (
    ExtractionRecipe,
    decode_video,
    infer_video_features,
    load_i3d,
    sliding_window_starts,
)
from elsc.resources import require_resources
from elsc.utils import atomic_json_dump, sha256_file


STREAMS = {
    "domain_agnostic": {
        "checkpoint": "artifacts/pretrained/bsl5k.pth.tar",
        "sha256": "6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f",
        "release_root": "artifacts/sign_features/ph_domain_agnostic/test",
        "gate_required": True,
        "checkpoint_scope": "released_domain_agnostic_bsl5k",
    },
    "domain_aware": {
        "checkpoint": "artifacts/pretrained/domain_aware_I3D_H2S.pth.tar",
        "sha256": "99e101d696ff63131b5d44fa6e465201216604ba5d8cc773f3cefa4a96ebd518",
        "release_root": "artifacts/sign_features/ph_domain_aware/test",
        "gate_required": False,
        "checkpoint_scope": "released_how2sign_transfer_not_phoenix_target_checkpoint",
    },
}


def _frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count < 1:
        raise ValueError(f"cannot determine frame count: {path}")
    return count


def _select_length_quantiles(video_root: Path, release_root: Path) -> list[Path]:
    release_ids = {path.stem for path in release_root.glob("*.pkl")}
    if not release_ids:
        raise FileNotFoundError(f"no release features found: {release_root}")
    missing = sorted(identifier for identifier in release_ids if not (video_root / f"{identifier}.mp4").is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} release-test videos are absent from {video_root}; first={missing[0]}"
        )
    values = sorted(
        (
            (_frame_count(video_root / f"{identifier}.mp4"), video_root / f"{identifier}.mp4")
            for identifier in release_ids
        ),
        key=lambda item: (item[0], item[1].name),
    )
    if not values:
        raise FileNotFoundError(f"no test videos found: {video_root}")
    indexes = (0, len(values) // 2, len(values) - 1)
    return [values[index][1] for index in indexes]


def _load_release(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    array = np.asarray(value["feature"] if isinstance(value, dict) else value)
    if array.ndim != 2 or array.shape[1] != 1024:
        raise ValueError(f"invalid release feature shape at {path}: {array.shape}")
    return np.asarray(array, dtype=np.float32)


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "passed": False,
        }
    difference = np.abs(actual - expected)
    numerator = np.sum(actual * expected, axis=1)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
    cosine = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
    result = {
        "shape_match": True,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "min_token_cosine": float(cosine.min()),
        "mean_token_cosine": float(cosine.mean()),
    }
    result["passed"] = (
        result["mean_abs_error"] <= 0.002
        and result["mean_token_cosine"] >= 0.9985
    )
    return result


def validate_release(
    video_root: Path,
    output: Path,
    device: torch.device,
    *,
    batch_size: int,
    min_free_disk_gib: float,
    min_free_gpu_gib: float,
) -> dict[str, Any]:
    recipe = ExtractionRecipe(stride=1)
    implementation = Path("third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py")
    resources = require_resources(
        output,
        device,
        min_disk_gib=min_free_disk_gib,
        min_gpu_gib=min_free_gpu_gib,
        operation="I3D-to-release feature parity validation",
    )
    samples = _select_length_quantiles(
        video_root, Path(STREAMS["domain_agnostic"]["release_root"])
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "test_feature_extraction_recipe_validation_only_not_model_selection",
        "selection_eligible": False,
        "gate_policy": {
            "required_streams": ["domain_agnostic"],
            "mean_abs_error_max": 0.002,
            "mean_token_cosine_min": 0.9985,
            "domain_aware_note": (
                "The published downloadable domain-aware checkpoint is trained for How2Sign, "
                "not the unavailable PHOENIX target-domain encoder. Its comparison is "
                "diagnostic only and local extraction uses it consistently across all splits."
            ),
        },
        "video_root": str(video_root.resolve()),
        "recipe": recipe.__dict__,
        "recipe_sha256": recipe.digest,
        "upstream_i3d_sha256": sha256_file(implementation),
        "resources": resources,
        "samples": [path.stem for path in samples],
        "streams": {},
    }
    all_passed = True
    for stream, settings in STREAMS.items():
        checkpoint = Path(settings["checkpoint"])
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != settings["sha256"]:
            raise ValueError(f"{stream} checkpoint SHA-256 mismatch")
        model = load_i3d(checkpoint, implementation, device)
        comparisons: dict[str, Any] = {}
        active_batch = batch_size
        for video in samples:
            frames, _fps = decode_video(video, recipe)
            starts = sliding_window_starts(len(frames), recipe.clip_frames, recipe.stride)
            actual, active_batch = infer_video_features(
                model, frames, starts, recipe, device, active_batch
            )
            expected = _load_release(Path(settings["release_root"]) / f"{video.stem}.pkl")
            comparisons[video.stem] = _comparison(actual, expected)
            if settings["gate_required"]:
                all_passed &= bool(comparisons[video.stem]["passed"])
        report["streams"][stream] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "release_root": str(Path(settings["release_root"]).resolve()),
            "effective_batch_size": active_batch,
            "gate_required": settings["gate_required"],
            "checkpoint_scope": settings["checkpoint_scope"],
            "comparisons": comparisons,
        }
        del model
        torch.cuda.empty_cache()
    report["status"] = "passed" if all_passed else "mismatch"
    atomic_json_dump(report, output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare locally re-extracted I3D features against the official PH release"
    )
    parser.add_argument(
        "--video-root",
        default="/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/test",
    )
    parser.add_argument(
        "--output", default="artifacts/parity/i3d_release_feature_parity.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-free-disk-gib", type=float, default=20.0)
    parser.add_argument("--min-free-gpu-gib", type=float, default=12.0)
    args = parser.parse_args(argv)
    result = validate_release(
        Path(args.video_root),
        Path(args.output),
        torch.device(args.device),
        batch_size=args.batch_size,
        min_free_disk_gib=args.min_free_disk_gib,
        min_free_gpu_gib=args.min_free_gpu_gib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
