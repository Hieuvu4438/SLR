from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from elsc.config import config_hash, load_config
from elsc.utils import atomic_json_dump, sha256_file


RECALL_K = (1, 5, 10)
SUMMARY_METRICS = ("R1", "R5", "R10", "MedianR", "MeanR")


class ReportContractError(ValueError):
    pass


def _load_run(run_dir: Path, split: str) -> dict[str, Any]:
    metrics_path = run_dir / "evaluation" / split / "metrics.json"
    config_path = run_dir / "resolved_config.yaml"
    if not metrics_path.is_file() or not config_path.is_file():
        raise ReportContractError(f"run lacks resolved config or {split} metrics: {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = load_config(config_path, validate=False)
    if metrics.get("score_orientation") != "video_x_text" or metrics.get("units") != "percent":
        raise ReportContractError(f"invalid evaluation contract: {metrics_path}")
    if split == "test" and "selection_lock" not in metrics:
        raise ReportContractError(
            f"test result has no validated dev-selection lock: {metrics_path}"
        )
    return {
        "run_dir": str(run_dir),
        "seed": int(config["seed"]),
        "experiment": config.get("experiment"),
        "config_hash": config_hash(config),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "metrics": metrics,
    }


def _query_map(metrics: dict[str, Any], direction: str) -> dict[str, dict[str, Any]]:
    records = metrics.get("per_query", {}).get(direction)
    if not isinstance(records, list):
        raise ReportContractError(f"metrics lack per-query {direction} records")
    result = {str(record["query_id"]): record for record in records}
    if len(result) != len(records):
        raise ReportContractError(f"duplicate {direction} query IDs")
    return result


def _aligned_group_differences(
    baseline: dict[str, Any], method: dict[str, Any]
) -> dict[str, dict[str, dict[int, list[float]]]]:
    if baseline.get("id_hashes") != method.get("id_hashes"):
        raise ReportContractError("baseline and method gallery ID hashes differ")
    if baseline.get("metric_kernel") != method.get("metric_kernel"):
        raise ReportContractError("baseline and method metric kernels differ")
    grouped: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for direction in ("V2T", "T2V"):
        left = _query_map(baseline, direction)
        right = _query_map(method, direction)
        if set(left) != set(right):
            raise ReportContractError(f"baseline and method {direction} query IDs differ")
        for query_id in sorted(left):
            base_record = left[query_id]
            method_record = right[query_id]
            if base_record.get("matched_positive_id") != method_record.get("matched_positive_id"):
                raise ReportContractError(f"positive mapping changed for {direction}:{query_id}")
            group_id = query_id if direction == "V2T" else str(base_record["matched_positive_id"])
            for k in RECALL_K:
                difference = float(int(method_record["rank"]) < k) - float(
                    int(base_record["rank"]) < k
                )
                grouped[group_id][direction][k].append(difference)
    return grouped


def _bootstrap_deltas(
    paired_runs: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    pairs = list(paired_runs)
    if samples < 100:
        raise ReportContractError("paired bootstrap requires at least 100 samples")
    groups_by_seed = [
        _aligned_group_differences(base["metrics"], method["metrics"]) for base, method in pairs
    ]
    rng = np.random.default_rng(seed)
    draws = {
        direction: {k: np.empty(samples, dtype=np.float64) for k in RECALL_K}
        for direction in ("V2T", "T2V", "MEAN")
    }
    for draw in range(samples):
        sampled_seed_indexes = rng.integers(0, len(groups_by_seed), size=len(groups_by_seed))
        seed_values = {direction: {k: [] for k in RECALL_K} for direction in ("V2T", "T2V", "MEAN")}
        for seed_index in sampled_seed_indexes:
            groups = groups_by_seed[int(seed_index)]
            group_ids = sorted(groups)
            sampled_groups = rng.integers(0, len(group_ids), size=len(group_ids))
            direction_values = {
                direction: {k: [] for k in RECALL_K} for direction in ("V2T", "T2V")
            }
            for group_index in sampled_groups:
                record = groups[group_ids[int(group_index)]]
                for direction in ("V2T", "T2V"):
                    for k in RECALL_K:
                        direction_values[direction][k].extend(record[direction][k])
            for k in RECALL_K:
                v2t = float(np.mean(direction_values["V2T"][k])) * 100.0
                t2v = float(np.mean(direction_values["T2V"][k])) * 100.0
                seed_values["V2T"][k].append(v2t)
                seed_values["T2V"][k].append(t2v)
                seed_values["MEAN"][k].append(0.5 * (v2t + t2v))
        for direction in draws:
            for k in RECALL_K:
                draws[direction][k][draw] = np.mean(seed_values[direction][k])
    result: dict[str, Any] = {}
    for direction, values_by_k in draws.items():
        result[direction] = {}
        for k, values in values_by_k.items():
            low, high = np.quantile(values, (0.025, 0.975))
            result[direction][f"R{k}"] = {
                "ci95": [float(low), float(high)],
                "bootstrap_mean": float(values.mean()),
            }
    return result


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for direction in ("V2T", "T2V"):
        result[direction] = {}
        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [float(run["metrics"][direction][metric]) for run in runs], dtype=np.float64
            )
            result[direction][metric] = {
                "mean": float(values.mean()),
                "std_across_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
                "values": values.tolist(),
            }
    return result


def compare_runs(
    baseline_dirs: Iterable[str | Path],
    method_dirs: Iterable[str | Path],
    *,
    split: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260907,
) -> dict[str, Any]:
    baseline = [_load_run(Path(path), split) for path in baseline_dirs]
    method = [_load_run(Path(path), split) for path in method_dirs]
    if not baseline or len(baseline) != len(method):
        raise ReportContractError("provide the same nonzero number of baseline and method runs")
    if [run["seed"] for run in baseline] != [run["seed"] for run in method]:
        raise ReportContractError("baseline and method seeds must be paired in the same order")
    for left, right in zip(baseline, method, strict=True):
        _aligned_group_differences(left["metrics"], right["metrics"])
    baseline_aggregate = _aggregate(baseline)
    method_aggregate = _aggregate(method)
    deltas: dict[str, Any] = {}
    for direction in ("V2T", "T2V"):
        deltas[direction] = {}
        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [
                    float(right["metrics"][direction][metric])
                    - float(left["metrics"][direction][metric])
                    for left, right in zip(baseline, method, strict=True)
                ]
            )
            deltas[direction][metric] = {
                "mean": float(values.mean()),
                "std_across_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
                "values": values.tolist(),
            }
    return {
        "schema_version": 1,
        "result_kind": "measured_local",
        "split": split,
        "units": "percentage_points",
        "paired_seeds": [run["seed"] for run in baseline],
        "baseline_runs": [{key: run[key] for key in run if key != "metrics"} for run in baseline],
        "method_runs": [{key: run[key] for key in run if key != "metrics"} for run in method],
        "baseline": baseline_aggregate,
        "method": method_aggregate,
        "delta": deltas,
        "paired_bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "resampling_unit": "video_group_with_seed_hierarchy",
            "rank_policy": "evaluator_scalar_per_query_rank",
            "note": (
                "Primary aggregates retain CiCo's direction-specific tie policies; "
                "paired confidence intervals require one scalar rank per query and therefore "
                "cannot reproduce T2V's duplicate-rank tie expansion."
            ),
            "delta_percentage_points": _bootstrap_deltas(
                zip(baseline, method, strict=True),
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
        },
        "test_used_for_tuning": False if split == "test" else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate paired measured runs and compute video-group bootstrap intervals"
    )
    parser.add_argument("--baseline-runs", nargs="+", required=True)
    parser.add_argument("--method-runs", nargs="+", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare_runs(
        args.baseline_runs,
        args.method_runs,
        split=args.split,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
