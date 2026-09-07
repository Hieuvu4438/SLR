from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from elsc.config import config_hash, load_config
from elsc.provenance import ProvenanceError, validate_test_lock
from elsc.utils import atomic_json_dump, ordered_hash, sha256_file, sha256_json


RECALL_K = (1, 5, 10)
SUMMARY_METRICS = ("R1", "R5", "R10", "MedianR", "MeanR")


class ReportContractError(ValueError):
    pass


TRAINING_SOURCE_PATHS = (
    "elsc/config.py",
    "elsc/data/cache_dataset.py",
    "elsc/data/cico_dataset.py",
    "elsc/data/manifest.py",
    "elsc/data/tokenize.py",
    "elsc/data/views.py",
    "elsc/data/word_offsets.py",
    "elsc/evaluate.py",
    "elsc/evaluation/cico_eval.py",
    "elsc/evaluation/runtime.py",
    "elsc/losses/caption.py",
    "elsc/losses/coarse.py",
    "elsc/losses/distillation.py",
    "elsc/losses/evidence.py",
    "elsc/losses/lexical.py",
    "elsc/mining/build_cache.py",
    "elsc/mining/negative_graph.py",
    "elsc/mining/teacher_align.py",
    "elsc/models/adapter.py",
    "elsc/models/local_head.py",
    "elsc/models/retriever.py",
    "elsc/provenance.py",
    "elsc/resources.py",
    "elsc/train.py",
    "elsc/upstream/cico_bridge.py",
    "elsc/upstream/factory.py",
    "elsc/utils.py",
)


def _implementation_source_contract(provenance: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    implementation = provenance.get("implementation")
    if not isinstance(implementation, dict):
        raise ReportContractError(f"run lacks implementation provenance: {run_dir}")
    if (
        implementation.get("status") != "ready"
        or implementation.get("tracked_worktree_dirty") is not False
    ):
        raise ReportContractError(f"run used unavailable or dirty tracked sources: {run_dir}")
    root_value = implementation.get("root")
    commit = implementation.get("commit")
    if not isinstance(root_value, str) or not isinstance(commit, str):
        raise ReportContractError(f"run implementation provenance is incomplete: {run_dir}")
    try:
        output = subprocess.run(
            [
                "git",
                "-C",
                root_value,
                "ls-tree",
                "-r",
                "--full-tree",
                commit,
                "--",
                *TRAINING_SOURCE_PATHS,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReportContractError(
            f"cannot resolve recorded implementation commit for {run_dir}"
        ) from error
    blobs: dict[str, str] = {}
    for line in output.splitlines():
        try:
            metadata, path = line.split("\t", 1)
            _, object_type, object_id = metadata.split()
        except ValueError as error:
            raise ReportContractError(f"invalid git source record for {run_dir}") from error
        if object_type != "blob":
            raise ReportContractError(f"non-file training source at {path}: {run_dir}")
        blobs[path] = object_id
    missing = sorted(set(TRAINING_SOURCE_PATHS) - set(blobs))
    if missing:
        raise ReportContractError(
            f"recorded commit lacks training source paths for {run_dir}: " + ", ".join(missing)
        )
    return {
        "schema_version": 1,
        "source_paths": list(TRAINING_SOURCE_PATHS),
        "git_blob_ids": blobs,
        "source_hash": sha256_json(blobs),
    }


def _evaluation_contract(config: dict[str, Any], dev_manifest_sha256: str) -> dict[str, Any]:
    required = {
        "upstream.cico_commit": config.get("upstream", {}).get("cico_commit"),
        "data.dataset": config.get("data", {}).get("dataset"),
        "data.caption_language": config.get("data", {}).get("caption_language"),
        "data.feature_dim": config.get("data", {}).get("feature_dim"),
        "data.feature_len": config.get("data", {}).get("feature_len"),
        "data.max_words": config.get("data", {}).get("max_words"),
        "data.combine_type": config.get("data", {}).get("combine_type"),
        "data.alpha": config.get("data", {}).get("alpha"),
        "data.feature_path_mode": config.get("data", {}).get("feature_path_mode"),
        "data.sampling": config.get("data", {}).get("sampling"),
        "data.text_augmentation": config.get("data", {}).get("text_augmentation"),
        "model.sim_header": config.get("model", {}).get("sim_header"),
        "model.dual_mix": config.get("model", {}).get("dual_mix"),
        "model.mix_design": config.get("model", {}).get("mix_design"),
        "evaluation.scorer": config.get("evaluation", {}).get("scorer"),
        "evaluation.metrics": config.get("evaluation", {}).get("metrics"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ReportContractError(
            "run config lacks controlled evaluation fields: " + ", ".join(missing)
        )
    evaluation = config.get("evaluation", {})
    if evaluation.get("full_gallery") is not True:
        raise ReportContractError("run config must explicitly enable full-gallery evaluation")
    if evaluation.get("filter_by_aux_eligibility") is not False:
        raise ReportContractError("run config must explicitly disable auxiliary gallery filtering")
    return {
        "schema_version": 1,
        "dev_manifest_sha256": dev_manifest_sha256,
        "fields": required,
        "full_gallery": True,
        "filter_by_aux_eligibility": False,
    }


def _training_control_contract(
    config: dict[str, Any], run_dir: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    train_manifest = config.get("data", {}).get("train_manifest")
    if not isinstance(train_manifest, str) or not Path(train_manifest).is_file():
        raise ReportContractError(f"run config has no readable train manifest: {run_dir}")
    adapter = config.get("model", {}).get("adapter", {})
    train = config.get("train", {})
    required = {
        "seed": config.get("seed"),
        "model.init_checkpoint_sha256": config.get("model", {}).get("init_checkpoint_sha256"),
        "model.backbone_frozen": config.get("model", {}).get("backbone_frozen"),
        "model.logit_scale_frozen": config.get("model", {}).get("logit_scale_frozen"),
        "model.adapter.enabled": adapter.get("enabled"),
        "model.adapter.radius": adapter.get("radius"),
        "model.adapter.hidden_dim": adapter.get("hidden_dim"),
        "model.adapter.zero_init_output": adapter.get("zero_init_output"),
        "model.adapter.trainable": adapter.get("trainable", True),
        "train.epochs": train.get("epochs"),
        "train.per_device_batch": train.get("per_device_batch"),
        "train.accumulation_steps": train.get("accumulation_steps"),
        "train.optimizer": train.get("optimizer"),
        "train.beta1": train.get("beta1"),
        "train.beta2": train.get("beta2"),
        "train.epsilon": train.get("epsilon"),
        "train.exclude_bias_and_1d_from_weight_decay": train.get(
            "exclude_bias_and_1d_from_weight_decay"
        ),
        "train.core_lr": train.get("core_lr"),
        "train.adapter_lr": train.get("adapter_lr"),
        "train.head_lr": train.get("head_lr"),
        "train.weight_decay": train.get("weight_decay"),
        "train.warmup_ratio": train.get("warmup_ratio"),
        "train.schedule": train.get("schedule"),
        "train.grad_clip_norm": train.get("grad_clip_norm"),
        "train.precision": train.get("precision"),
        "train.checkpoint_metric": train.get("checkpoint_metric"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ReportContractError("run config lacks matched-training fields: " + ", ".join(missing))
    initialization = provenance.get("initialization")
    if (
        not isinstance(initialization, dict)
        or initialization.get("sha256") != required["model.init_checkpoint_sha256"]
    ):
        raise ReportContractError(f"run initialization provenance differs from config: {run_dir}")
    method = str(config.get("method"))
    teacher_sha256 = None
    cache_meta_sha256 = None
    implementation_contract = None
    if method != "baseline":
        implementation_contract = _implementation_source_contract(provenance, run_dir)
        teacher_sha256 = config.get("model", {}).get("teacher_checkpoint_sha256")
        teacher = provenance.get("teacher")
        if (
            not isinstance(teacher_sha256, str)
            or not isinstance(teacher, dict)
            or teacher.get("sha256") != teacher_sha256
        ):
            raise ReportContractError(f"run teacher provenance differs from config: {run_dir}")
        cache_path = config.get("cache", {}).get("path")
        cache_meta = Path(str(cache_path)) / "cache_meta.json"
        if not cache_meta.is_file():
            raise ReportContractError(f"run cache metadata is missing: {cache_meta}")
        cache_meta_sha256 = sha256_file(cache_meta)
        if provenance.get("cache_meta_sha256") != cache_meta_sha256:
            raise ReportContractError(f"run cache provenance is stale: {run_dir}")
    return {
        "schema_version": 1,
        "train_manifest_sha256": sha256_file(train_manifest),
        "teacher_checkpoint_sha256": teacher_sha256,
        "cache_meta_sha256": cache_meta_sha256,
        "implementation_source_hash": (
            implementation_contract["source_hash"] if implementation_contract is not None else None
        ),
        "fields": required,
    }


def _validate_summary_from_ranks(
    metrics: dict[str, Any], direction: str, ranks: list[int], path: Path
) -> None:
    if not ranks:
        raise ReportContractError(f"{direction} metrics have no primary ranks: {path}")
    array = np.asarray(ranks, dtype=np.int64)
    expected = {
        "R1": float((array < 1).mean() * 100.0),
        "R5": float((array < 5).mean() * 100.0),
        "R10": float((array < 10).mean() * 100.0),
        "MedianR": float(np.median(array) + 1),
        "MeanR": float(np.mean(array) + 1),
    }
    summary = metrics.get(direction)
    if not isinstance(summary, dict):
        raise ReportContractError(f"metrics lack {direction} summary: {path}")
    for name, expected_value in expected.items():
        try:
            actual = float(summary[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ReportContractError(f"metrics lack numeric {direction} {name}: {path}") from error
        if not np.isfinite(actual) or not np.isclose(actual, expected_value, rtol=0.0, atol=1e-9):
            raise ReportContractError(f"{direction} {name} does not match per-query ranks: {path}")
    if summary.get("cols") != ranks:
        raise ReportContractError(f"{direction} rank vector differs from per-query ranks: {path}")


def _validate_full_gallery_metrics(metrics: dict[str, Any], path: Path) -> None:
    if metrics.get("schema_version") != 1:
        raise ReportContractError(f"unsupported metrics schema: {path}")
    if metrics.get("metric_kernel") not in {
        "cico_direction_specific_singleton_tie_behavior",
        "id_multi_positive_best_rank",
    }:
        raise ReportContractError(f"missing or unsupported metric kernel: {path}")
    gallery = metrics.get("gallery")
    if not isinstance(gallery, dict):
        raise ReportContractError(f"metrics lack full-gallery cardinalities: {path}")
    try:
        video_count = int(gallery["videos"])
        text_count = int(gallery["texts"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReportContractError(f"invalid full-gallery cardinalities: {path}") from error
    if video_count < 1 or text_count < 1:
        raise ReportContractError(f"full gallery must be non-empty: {path}")
    id_hashes = metrics.get("id_hashes")
    if not isinstance(id_hashes, dict) or any(
        not isinstance(id_hashes.get(name), str) for name in ("videos", "texts", "positive_mapping")
    ):
        raise ReportContractError(f"metrics lack gallery ID hashes: {path}")

    direction_contracts = {
        "V2T": (video_count, text_count, "videos"),
        "T2V": (text_count, video_count, "texts"),
    }
    query_ids_by_direction: dict[str, list[str]] = {}
    candidate_ids_by_direction: dict[str, set[str]] = {}
    primary_ranks_by_direction: dict[str, list[int]] = {}
    for direction, (query_count, candidate_count, hash_name) in direction_contracts.items():
        records = metrics.get("per_query", {}).get(direction)
        if not isinstance(records, list) or len(records) != query_count:
            raise ReportContractError(
                f"{direction} query count does not match the full gallery: {path}"
            )
        query_ids = [str(record.get("query_id")) for record in records]
        if len(set(query_ids)) != query_count:
            raise ReportContractError(f"duplicate {direction} query IDs: {path}")
        if ordered_hash(query_ids) != id_hashes[hash_name]:
            raise ReportContractError(
                f"{direction} ordered query IDs do not match their gallery hash: {path}"
            )
        expected_candidates: set[str] | None = None
        query_ranks: list[int] = []
        official_tie_ranks: list[int] = []
        for record in records:
            ranked = record.get("ranked_candidate_ids")
            if (
                not isinstance(ranked, list)
                or len(ranked) != candidate_count
                or len({str(value) for value in ranked}) != candidate_count
            ):
                raise ReportContractError(
                    f"{direction} ranked candidates do not contain the full gallery: {path}"
                )
            candidates = {str(value) for value in ranked}
            if expected_candidates is None:
                expected_candidates = candidates
            elif candidates != expected_candidates:
                raise ReportContractError(
                    f"{direction} candidate set changes between queries: {path}"
                )
            try:
                rank = int(record["rank"])
            except (KeyError, TypeError, ValueError) as error:
                raise ReportContractError(f"invalid {direction} query rank: {path}") from error
            if not 0 <= rank < candidate_count:
                raise ReportContractError(f"out-of-range {direction} query rank: {path}")
            query_ranks.append(rank)
            ties = record.get("official_tie_ranks")
            if not isinstance(ties, list) or not ties:
                raise ReportContractError(f"{direction} query lacks official tie ranks: {path}")
            try:
                numeric_ties = [int(value) for value in ties]
            except (TypeError, ValueError) as error:
                raise ReportContractError(
                    f"invalid {direction} official tie ranks: {path}"
                ) from error
            if any(not 0 <= value < candidate_count for value in numeric_ties):
                raise ReportContractError(f"out-of-range {direction} official tie rank: {path}")
            official_tie_ranks.extend(numeric_ties)
            if str(record.get("matched_positive_id")) not in candidates:
                raise ReportContractError(
                    f"{direction} matched positive is absent from the gallery: {path}"
                )
        query_ids_by_direction[direction] = query_ids
        candidate_ids_by_direction[direction] = expected_candidates or set()
        primary_ranks_by_direction[direction] = (
            official_tie_ranks
            if metrics["metric_kernel"] == "cico_direction_specific_singleton_tie_behavior"
            and direction == "T2V"
            else query_ranks
        )

    if candidate_ids_by_direction["V2T"] != set(query_ids_by_direction["T2V"]):
        raise ReportContractError(f"V2T candidate IDs differ from T2V query IDs: {path}")
    if candidate_ids_by_direction["T2V"] != set(query_ids_by_direction["V2T"]):
        raise ReportContractError(f"T2V candidate IDs differ from V2T query IDs: {path}")
    if (
        metrics["metric_kernel"] == "cico_direction_specific_singleton_tie_behavior"
        and video_count != text_count
    ):
        raise ReportContractError(f"singleton CiCo metrics require a square gallery: {path}")
    for direction, ranks in primary_ranks_by_direction.items():
        _validate_summary_from_ranks(metrics, direction, ranks, path)


def _load_run(run_dir: Path, split: str) -> dict[str, Any]:
    metrics_path = run_dir / "evaluation" / split / "metrics.json"
    config_path = run_dir / "resolved_config.yaml"
    selection_path = run_dir / "selection.json"
    if not metrics_path.is_file() or not config_path.is_file() or not selection_path.is_file():
        raise ReportContractError(
            f"run lacks resolved config, dev selection, or {split} metrics: {run_dir}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = load_config(config_path, validate=False)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.is_file():
        raise ReportContractError(f"run lacks training provenance: {run_dir}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    relative_checkpoint = selection.get("checkpoint")
    if not isinstance(relative_checkpoint, str):
        raise ReportContractError(f"selection lacks checkpoint path: {selection_path}")
    checkpoint = run_dir / relative_checkpoint
    try:
        validate_test_lock(selection_path, checkpoint, run_dir=run_dir, config=config)
    except (ProvenanceError, OSError, ValueError) as error:
        raise ReportContractError(f"invalid frozen run selection: {run_dir}: {error}") from error
    if metrics.get("score_orientation") != "video_x_text" or metrics.get("units") != "percent":
        raise ReportContractError(f"invalid evaluation contract: {metrics_path}")
    if metrics.get("split") != split:
        raise ReportContractError(f"metrics split does not match requested split: {metrics_path}")
    _validate_full_gallery_metrics(metrics, metrics_path)
    metric_checkpoint = metrics.get("checkpoint")
    if not isinstance(metric_checkpoint, dict) or metric_checkpoint.get("sha256") != selection.get(
        "checkpoint_sha256"
    ):
        raise ReportContractError(
            f"metrics were not produced from the dev-selected checkpoint: {metrics_path}"
        )
    if split == "test" and "selection_lock" not in metrics:
        raise ReportContractError(
            f"test result has no validated dev-selection lock: {metrics_path}"
        )
    contract = _evaluation_contract(config, str(selection["dev_manifest_sha256"]))
    training_contract = _training_control_contract(config, run_dir, provenance)
    return {
        "run_dir": str(run_dir),
        "seed": int(config["seed"]),
        "experiment": config.get("experiment"),
        "config_hash": config_hash(config),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "selection_sha256": sha256_file(selection_path),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "evaluation_contract": contract,
        "evaluation_contract_hash": sha256_json(contract),
        "training_control_contract": training_contract,
        "training_control_contract_hash": sha256_json(training_contract),
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
        if left["evaluation_contract_hash"] != right["evaluation_contract_hash"]:
            raise ReportContractError("baseline and method controlled evaluation contracts differ")
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
