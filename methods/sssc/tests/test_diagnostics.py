from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from method1.diagnostics import _distribution, _support_metrics
from method1.utils import atomic_jsonl_dump


def test_support_diagnostic_matches_shared_independent_fixture() -> None:
    result = _support_metrics(
        np.eye(2, dtype=np.float32),
        np.array([True, True]),
        [10, 20],
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        tau=0.07,
        margin=0.05,
    )
    assert result["shared_margin"] == pytest.approx(1.0, abs=2e-6)
    assert result["independent_margin"] == pytest.approx(0.0, abs=2e-6)
    assert result["peak_displacement_feature_rows"] == 10
    assert result["support_overlap"] < 1e-5
    assert result["support_jsd"] == pytest.approx(np.log(2.0), abs=2e-5)


def test_diagnostic_distribution_and_atomic_records(tmp_path: Path) -> None:
    assert _distribution([])["mean"] is None
    summary = _distribution([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    path = tmp_path / "records.jsonl"
    atomic_jsonl_dump([{"id": 1}, {"id": 2}], path)
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == [1, 2]
