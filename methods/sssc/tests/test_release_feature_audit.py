from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.audit_ph_release_features import FeatureAlignmentError, audit_alignment


def _write_feature(root: Path, name: str, rows: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / name).open("wb") as handle:
        pickle.dump({"feature": rows.astype(np.float32)}, handle)


def test_release_feature_audit_reports_aligned_streams(tmp_path: Path) -> None:
    roots = {label: tmp_path / label for label in ("la", "lw", "ra", "rw")}
    first = np.zeros((2, 1024), dtype=np.float32)
    first[:, 0] = 1.0
    second = np.zeros((2, 1024), dtype=np.float32)
    second[:, 1] = 1.0
    for label in ("la", "ra"):
        _write_feature(roots[label], "sample.pkl", first)
    for label in ("lw", "rw"):
        _write_feature(roots[label], "sample.pkl", second)
    report = audit_alignment(
        local_agnostic=roots["la"],
        local_aware=roots["lw"],
        release_agnostic=roots["ra"],
        release_aware=roots["rw"],
        agnostic_weight=0.9,
        expected_files=1,
        release_archive=None,
    )
    assert report["file_count"] == 1
    assert report["row_count"] == 2
    assert report["row_cosine"]["configured_mixture"]["mean"] == pytest.approx(1.0)
    assert report["per_video_relative_l2"]["configured_mixture"]["mean"] == 0.0
    assert len(report["content_sha256"]) == 64


def test_release_feature_audit_rejects_filename_set_mismatch(tmp_path: Path) -> None:
    roots = {label: tmp_path / label for label in ("la", "lw", "ra", "rw")}
    rows = np.ones((1, 1024), dtype=np.float32)
    for label in roots:
        _write_feature(roots[label], "sample.pkl", rows)
    _write_feature(roots["la"], "extra.pkl", rows)
    with pytest.raises(FeatureAlignmentError, match="filename sets differ"):
        audit_alignment(
            local_agnostic=roots["la"],
            local_aware=roots["lw"],
            release_agnostic=roots["ra"],
            release_aware=roots["rw"],
            agnostic_weight=0.9,
            expected_files=None,
            release_archive=None,
        )
