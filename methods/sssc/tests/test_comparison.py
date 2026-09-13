from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from method1.comparison import ComparisonError, compare_runs


def _write_run(root: Path, arm: str, ranks: list[int], artifacts=None) -> None:
    root.mkdir(parents=True)
    (root / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    metrics = {
        "query_ids": {"T2V": ["g0", "g1"], "V2T": ["v0", "v1"]},
        "T2V": {
            "R1": 50.0,
            "R5": 100.0,
            "R10": 100.0,
            "MedR": 1.5,
            "MeanR": 1.5,
            "MRR": 0.75,
            "ranks": ranks,
        },
        "V2T": {
            "R1": 50.0,
            "R5": 100.0,
            "R10": 100.0,
            "MedR": 1.5,
            "MeanR": 1.5,
            "MRR": 0.75,
            "ranks": ranks,
        },
    }
    torch.save(
        {
            "training_run_complete": True,
            "artifact_hashes": artifacts or {"teacher": "same", "cache": "same"},
            "implementation_revision": "abc123",
            "resolved_config": {
                "seed": 42,
                "auxiliary": {"arm": arm, "support_mode": arm},
                "output": {"root": str(root)},
            },
            "arm": arm,
            "dev_metrics": metrics,
        },
        root / "best_dev.pt",
    )


def test_compare_runs_checks_resources_and_paired_rank_changes(tmp_path: Path) -> None:
    left = tmp_path / "span_independent" / "seed42"
    right = tmp_path / "span_shared" / "seed42"
    _write_run(left, "span_independent", [2, 1])
    _write_run(right, "span_shared", [1, 2])
    report = compare_runs(left, right)
    assert report["paired_resources_verified"] is True
    assert report["paired_rank_changes"]["T2V"] == {
        "query_count": 2,
        "improved_in_run_b": 1,
        "regressed_in_run_b": 1,
        "unchanged": 0,
        "mean_rank_change_b_minus_a": 0.0,
        "median_rank_change_b_minus_a": 0.0,
    }
    assert "auxiliary.arm" in report["config_differences"]
    saved = json.loads(Path(report["output"]).read_text(encoding="utf-8"))
    assert saved["status"] == "complete"


def test_compare_runs_fails_on_unmatched_resources(tmp_path: Path) -> None:
    left = tmp_path / "a"
    right = tmp_path / "b"
    _write_run(left, "span_independent", [1, 2])
    _write_run(right, "span_shared", [1, 2], artifacts={"teacher": "different"})
    with pytest.raises(ComparisonError, match="resource"):
        compare_runs(left, right)
