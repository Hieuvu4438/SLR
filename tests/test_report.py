from __future__ import annotations

import json
from pathlib import Path

import pytest

from elsc.config import config_hash, load_config
from elsc.report import ReportContractError, compare_runs
from elsc.utils import sha256_file


def _write_run(root: Path, seed: int, ranks: tuple[list[int], list[int]], id_hash: str) -> Path:
    root.mkdir(parents=True)
    dev_manifest = root / "dev.jsonl"
    dev_manifest.write_text("{}\n", encoding="utf-8")
    (root / "resolved_config.yaml").write_text(
        f"schema_version: 1\nseed: {seed}\ndata:\n  dev_manifest: {dev_manifest}\n",
        encoding="utf-8",
    )
    checkpoint = root / "checkpoints" / "best_dev.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"selected checkpoint")
    resolved = load_config(root / "resolved_config.yaml", validate=False)
    selection = {
        "selection_split": "dev",
        "test_used_for_selection": False,
        "checkpoint": "checkpoints/best_dev.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_hash": config_hash(resolved),
        "dev_manifest_sha256": sha256_file(dev_manifest),
    }
    (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    output = root / "evaluation" / "dev"
    output.mkdir(parents=True)
    directions = {}
    ids = ["v0", "v1", "v2"]
    for direction, values in zip(("V2T", "T2V"), ranks, strict=True):
        directions[direction] = {
            "R1": 100.0 * sum(rank < 1 for rank in values) / 3,
            "R5": 100.0 * sum(rank < 5 for rank in values) / 3,
            "R10": 100.0 * sum(rank < 10 for rank in values) / 3,
            "MedianR": 1.0,
            "MeanR": 1.0 + sum(values) / 3,
        }
        query_ids = ids if direction == "V2T" else ["t0", "t1", "t2"]
        directions.setdefault("per_query", {})[direction] = [
            {
                "query_id": query,
                "rank": rank,
                "matched_positive_id": f"t{index}" if direction == "V2T" else ids[index],
            }
            for index, (query, rank) in enumerate(zip(query_ids, values, strict=True))
        ]
    metrics = {
        "schema_version": 1,
        "score_orientation": "video_x_text",
        "units": "percent",
        "split": "dev",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "id_hashes": {"videos": id_hash, "texts": id_hash},
        **directions,
    }
    (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return root


def test_paired_report_uses_query_identity_and_video_groups(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([1, 0, 2], [1, 0, 2]), "same")
    method = _write_run(tmp_path / "method", 42, ([0, 0, 2], [0, 0, 2]), "same")
    result = compare_runs(
        [baseline], [method], split="dev", bootstrap_samples=200, bootstrap_seed=7
    )
    assert result["result_kind"] == "measured_local"
    assert result["delta"]["V2T"]["R1"]["mean"] == pytest.approx(100 / 3)
    mean_ci = result["paired_bootstrap"]["delta_percentage_points"]["MEAN"]["R1"]
    assert mean_ci["ci95"][0] <= 100 / 3 <= mean_ci["ci95"][1]


def test_paired_report_rejects_different_galleries(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]), "left")
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]), "right")
    with pytest.raises(ReportContractError, match="gallery ID hashes differ"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)


def test_paired_report_rejects_metrics_from_nonselected_checkpoint(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]), "same")
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]), "same")
    metrics_path = method / "evaluation" / "dev" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["checkpoint"]["sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ReportContractError, match="dev-selected checkpoint"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)
