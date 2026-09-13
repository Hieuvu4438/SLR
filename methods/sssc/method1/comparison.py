from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .utils import atomic_json_dump, sha256_file, stable_seed


class ComparisonError(RuntimeError):
    pass


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
        return output
    return {prefix: value}


def _load_run(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ComparisonError(f"run must be a directory: {root}")
    checkpoint_path = root / "best_dev.pt"
    manifest_path = root / "run_manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise ComparisonError(f"run lacks best_dev.pt or run_manifest.json: {root}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise ComparisonError(f"cannot read run checkpoint {checkpoint_path}: {error}") from error
    if not checkpoint.get("training_run_complete", False):
        raise ComparisonError(f"run is incomplete or only a pilot: {root}")
    metrics = checkpoint.get("dev_metrics")
    if not isinstance(metrics, dict):
        raise ComparisonError(f"selected checkpoint has no dev metrics: {root}")
    return {
        "root": root,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint": checkpoint,
        "metrics": metrics,
    }


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        direction: {
            key: float(metrics[direction][key])
            for key in ("R1", "R5", "R10", "MedR", "MeanR", "MRR")
        }
        for direction in ("T2V", "V2T")
    }


def _paired_cluster_bootstrap(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    cluster_ids: list[str],
    *,
    seed: int,
    iterations: int = 10_000,
) -> dict[str, Any]:
    if iterations < 1 or ranks_a.shape != ranks_b.shape or len(cluster_ids) != len(ranks_a):
        raise ComparisonError("invalid paired cluster-bootstrap inputs")
    clusters: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        clusters.setdefault(str(cluster_id), []).append(index)
    ordered_clusters = list(clusters)
    generator = np.random.default_rng(seed)
    samples = {key: np.empty(iterations, dtype=np.float64) for key in ("R1", "R5", "R10", "MRR")}
    for iteration in range(iterations):
        selected_clusters = generator.choice(ordered_clusters, len(ordered_clusters), replace=True)
        indexes = np.concatenate(
            [np.asarray(clusters[str(cluster)], dtype=np.int64) for cluster in selected_clusters]
        )
        selected_a = ranks_a[indexes]
        selected_b = ranks_b[indexes]
        samples["R1"][iteration] = 100.0 * np.mean(selected_b <= 1) - 100.0 * np.mean(
            selected_a <= 1
        )
        samples["R5"][iteration] = 100.0 * np.mean(selected_b <= 5) - 100.0 * np.mean(
            selected_a <= 5
        )
        samples["R10"][iteration] = 100.0 * np.mean(
            selected_b <= 10
        ) - 100.0 * np.mean(selected_a <= 10)
        samples["MRR"][iteration] = np.mean(1.0 / selected_b) - np.mean(1.0 / selected_a)
    return {
        "iterations": iterations,
        "resampling_unit": "retrieval_group",
        "cluster_count": len(ordered_clusters),
        "confidence_level": 0.95,
        "delta_run_b_minus_a": {
            key: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for key, values in samples.items()
        },
    }


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    left = _load_run(run_a)
    right = _load_run(run_b)
    checkpoint_a = left["checkpoint"]
    checkpoint_b = right["checkpoint"]
    if checkpoint_a.get("artifact_hashes") != checkpoint_b.get("artifact_hashes"):
        raise ComparisonError("paired runs have different resource/training-protocol hashes")
    if checkpoint_a.get("implementation_revision") != checkpoint_b.get(
        "implementation_revision"
    ):
        raise ComparisonError("paired runs used different implementation revisions")
    config_a = _flatten(checkpoint_a.get("resolved_config", {}))
    config_b = _flatten(checkpoint_b.get("resolved_config", {}))
    config_differences = {
        key: {"run_a": config_a.get(key), "run_b": config_b.get(key)}
        for key in sorted(set(config_a) | set(config_b))
        if config_a.get(key) != config_b.get(key)
    }
    metrics_a = left["metrics"]
    metrics_b = right["metrics"]
    paired_rank_changes = {}
    paired_bootstrap = {}
    for direction in ("T2V", "V2T"):
        ranks_a = np.asarray(metrics_a[direction]["ranks"], dtype=np.int64)
        ranks_b = np.asarray(metrics_b[direction]["ranks"], dtype=np.int64)
        query_ids_a = metrics_a.get("query_ids", {}).get(direction)
        query_ids_b = metrics_b.get("query_ids", {}).get(direction)
        if ranks_a.shape != ranks_b.shape or query_ids_a != query_ids_b:
            raise ComparisonError(f"{direction} query identities/rank shapes differ")
        query_groups_a = metrics_a.get("query_group_ids", {}).get(direction)
        query_groups_b = metrics_b.get("query_group_ids", {}).get(direction)
        if query_groups_a != query_groups_b or not isinstance(query_groups_a, list):
            raise ComparisonError(f"{direction} paired query-group identities differ or are absent")
        delta = ranks_b - ranks_a
        paired_rank_changes[direction] = {
            "query_count": len(ranks_a),
            "improved_in_run_b": int(np.sum(delta < 0)),
            "regressed_in_run_b": int(np.sum(delta > 0)),
            "unchanged": int(np.sum(delta == 0)),
            "mean_rank_change_b_minus_a": float(delta.mean()),
            "median_rank_change_b_minus_a": float(np.median(delta)),
        }
        paired_bootstrap[direction] = _paired_cluster_bootstrap(
            ranks_a,
            ranks_b,
            query_groups_a,
            seed=stable_seed(
                checkpoint_a.get("resolved_config", {}).get("seed"),
                direction,
                checkpoint_a.get("arm"),
                checkpoint_b.get("arm"),
                "paired_cluster_bootstrap_v1",
            ),
        )
    summary_a = _metric_summary(metrics_a)
    summary_b = _metric_summary(metrics_b)
    metric_delta = {
        direction: {
            key: summary_b[direction][key] - summary_a[direction][key]
            for key in summary_a[direction]
        }
        for direction in ("T2V", "V2T")
    }
    arm_a = str(checkpoint_a.get("arm"))
    arm_b = str(checkpoint_b.get("arm"))
    seed = checkpoint_a.get("resolved_config", {}).get("seed")
    if seed != checkpoint_b.get("resolved_config", {}).get("seed"):
        raise ComparisonError("paired comparison requires the same experiment seed")
    report = {
        "schema_version": 1,
        "status": "complete",
        "paired_resources_verified": True,
        "seed": seed,
        "run_a": {
            "path": str(left["root"]),
            "arm": arm_a,
            "checkpoint_sha256": left["checkpoint_sha256"],
            "metrics": summary_a,
        },
        "run_b": {
            "path": str(right["root"]),
            "arm": arm_b,
            "checkpoint_sha256": right["checkpoint_sha256"],
            "metrics": summary_b,
        },
        "metric_delta_run_b_minus_a": metric_delta,
        "paired_rank_changes": paired_rank_changes,
        "paired_cluster_bootstrap": paired_bootstrap,
        "config_differences": config_differences,
        "implementation_revision": checkpoint_a.get("implementation_revision"),
        "artifact_hashes": checkpoint_a.get("artifact_hashes"),
    }
    common = Path(os.path.commonpath((left["root"], right["root"])))
    output = common / "comparisons" / f"seed{seed}_{arm_a}_vs_{arm_b}.json"
    atomic_json_dump(report, output)
    report["output"] = str(output.resolve())
    return report
