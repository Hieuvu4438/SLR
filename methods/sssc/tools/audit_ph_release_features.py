#!/usr/bin/env python3
"""Compare the active PH feature streams with the official CiCo test release."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from method1.utils import atomic_json_dump, sha256_file, sha256_json


class FeatureAlignmentError(ValueError):
    pass


def _load_feature(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - all roots are explicitly audited inputs
    if isinstance(value, dict):
        value = value.get("feature")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 1024 or not np.isfinite(array).all():
        raise FeatureAlignmentError(f"invalid feature array at {path}: {array.shape}")
    return array


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise FeatureAlignmentError("cannot summarize empty or nonfinite measurements")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return (left * right).sum(axis=1) / np.maximum(denominator, 1e-12)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1e-12))


def audit_alignment(
    *,
    local_agnostic: Path,
    local_aware: Path,
    release_agnostic: Path,
    release_aware: Path,
    agnostic_weight: float,
    expected_files: int | None,
    release_archive: Path | None,
) -> dict[str, Any]:
    roots = {
        "local_agnostic": local_agnostic,
        "local_aware": local_aware,
        "release_agnostic": release_agnostic,
        "release_aware": release_aware,
    }
    names_by_root = {
        label: {path.name for path in root.glob("*.pkl")} for label, root in roots.items()
    }
    reference_names = names_by_root["release_agnostic"]
    if not reference_names or any(names != reference_names for names in names_by_root.values()):
        counts = {label: len(names) for label, names in names_by_root.items()}
        raise FeatureAlignmentError(f"feature filename sets differ: {counts}")
    if expected_files is not None and len(reference_names) != expected_files:
        raise FeatureAlignmentError(
            f"found {len(reference_names)} aligned files, expected {expected_files}"
        )
    if not 0.0 <= agnostic_weight <= 1.0:
        raise FeatureAlignmentError("agnostic weight must be in [0,1]")

    row_cosines = {"agnostic": [], "aware": [], "configured_mixture": []}
    relative_l2 = {"agnostic": [], "aware": [], "configured_mixture": []}
    total_rows = 0
    for name in sorted(reference_names):
        arrays = {label: _load_feature(root / name) for label, root in roots.items()}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise FeatureAlignmentError(f"stream shapes differ for {name}: {sorted(shapes)}")
        total_rows += arrays["local_agnostic"].shape[0]
        pairs = {
            "agnostic": (arrays["local_agnostic"], arrays["release_agnostic"]),
            "aware": (arrays["local_aware"], arrays["release_aware"]),
        }
        pairs["configured_mixture"] = (
            agnostic_weight * arrays["local_agnostic"]
            + (1.0 - agnostic_weight) * arrays["local_aware"],
            agnostic_weight * arrays["release_agnostic"]
            + (1.0 - agnostic_weight) * arrays["release_aware"],
        )
        for label, (left, right) in pairs.items():
            row_cosines[label].extend(_row_cosine(left, right).tolist())
            relative_l2[label].append(_relative_l2(left, right))

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "dataset": "ph",
        "split": "test",
        "role": "resource_alignment_audit_not_retrieval_evaluation",
        "file_count": len(reference_names),
        "row_count": total_rows,
        "feature_dimension": 1024,
        "all_stream_filenames_and_shapes_match": True,
        "agnostic_weight": agnostic_weight,
        "roots": {label: str(root.resolve()) for label, root in roots.items()},
        "row_cosine": {label: _summary(values) for label, values in row_cosines.items()},
        "per_video_relative_l2": {
            label: _summary(values) for label, values in relative_l2.items()
        },
    }
    if release_archive is not None:
        if not release_archive.is_file():
            raise FeatureAlignmentError(f"release archive is missing: {release_archive}")
        report["release_archive"] = {
            "path": str(release_archive.resolve()),
            "bytes": release_archive.stat().st_size,
            "sha256": sha256_file(release_archive),
        }
    report["content_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-agnostic", type=Path, required=True)
    parser.add_argument("--local-aware", type=Path, required=True)
    parser.add_argument("--release-agnostic", type=Path, required=True)
    parser.add_argument("--release-aware", type=Path, required=True)
    parser.add_argument("--release-archive", type=Path)
    parser.add_argument("--agnostic-weight", type=float, default=0.9)
    parser.add_argument("--expected-files", type=int, default=642)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_alignment(
        local_agnostic=args.local_agnostic,
        local_aware=args.local_aware,
        release_agnostic=args.release_agnostic,
        release_aware=args.release_aware,
        agnostic_weight=args.agnostic_weight,
        expected_files=args.expected_files,
        release_archive=args.release_archive,
    )
    atomic_json_dump(report, args.output)
    print(args.output)
    print(report["content_sha256"])


if __name__ == "__main__":
    main()
