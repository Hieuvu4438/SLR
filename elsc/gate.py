from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elsc.config import config_hash, load_config
from elsc.utils import atomic_json_dump, sha256_file


class GateContractError(ValueError):
    pass


def _validate_report_header(report: dict[str, Any], *, gate: str) -> list[int]:
    if report.get("schema_version") != 1:
        raise GateContractError("comparison report must use schema_version=1")
    if report.get("result_kind") != "measured_local":
        raise GateContractError(f"Gate {gate} requires measured_local results")
    if report.get("split") != "dev":
        raise GateContractError(f"Gate {gate} may only use the dev split")
    if report.get("units") != "percentage_points":
        raise GateContractError(f"Gate {gate} requires percentage-point deltas")
    paired_seeds = report.get("paired_seeds")
    if (
        not isinstance(paired_seeds, list)
        or not paired_seeds
        or not all(isinstance(seed, int) for seed in paired_seeds)
        or len(set(paired_seeds)) != len(paired_seeds)
    ):
        raise GateContractError(f"Gate {gate} requires unique paired integer seeds")
    if report.get("test_used_for_tuning") is True:
        raise GateContractError(f"Gate {gate} refuses reports that used test for tuning")
    return paired_seeds


def evaluate_gain_gate(
    report: dict[str, Any],
    *,
    mean_r1_gain_min: float = 0.5,
    max_direction_r1_drop: float = 0.5,
) -> dict[str, Any]:
    paired_seeds = _validate_report_header(report, gate="G")
    try:
        directional = {
            direction: float(report["delta"][direction]["R1"]["mean"])
            for direction in ("T2V", "V2T")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise GateContractError("comparison report lacks directional R1 deltas") from error
    mean_gain = 0.5 * (directional["T2V"] + directional["V2T"])
    mean_pass = mean_gain >= float(mean_r1_gain_min)
    directions_pass = all(
        delta >= -float(max_direction_r1_drop) for delta in directional.values()
    )
    return {
        "schema_version": 1,
        "gate": "G",
        "status": "passed" if mean_pass and directions_pass else "no_go",
        "split": "dev",
        "paired_seeds": paired_seeds,
        "observed": {
            "mean_r1_gain_percentage_points": mean_gain,
            "directional_r1_delta_percentage_points": directional,
        },
        "criteria": {
            "mean_r1_gain_min_percentage_points": float(mean_r1_gain_min),
            "max_direction_r1_drop_percentage_points": float(max_direction_r1_drop),
            "mean_gain_pass": mean_pass,
            "directions_pass": directions_pass,
        },
        "test_used_for_gate": False,
    }


def _method_run_identity(report: dict[str, Any]) -> list[tuple[Any, ...]]:
    runs = report.get("method_runs")
    if not isinstance(runs, list) or not runs:
        raise GateContractError("Gate M requires method-run provenance")
    try:
        return [
            (
                int(run["seed"]),
                str(run["config_hash"]),
                str(run["selection_sha256"]),
                str(run["checkpoint_sha256"]),
                str(run["metrics_sha256"]),
            )
            for run in runs
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise GateContractError("Gate M method-run provenance is incomplete") from error


def _per_seed_mean_r1_delta(
    report: dict[str, Any], paired_seeds: list[int]
) -> list[float]:
    try:
        directional = [
            [float(value) for value in report["delta"][direction]["R1"]["values"]]
            for direction in ("T2V", "V2T")
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise GateContractError("Gate M report lacks per-seed directional R1 deltas") from error
    if any(len(values) != len(paired_seeds) for values in directional):
        raise GateContractError("Gate M per-seed deltas do not align with paired seeds")
    return [0.5 * (directional[0][index] + directional[1][index]) for index in range(len(paired_seeds))]


def evaluate_mechanism_gate(
    true_vs_random_report: dict[str, Any],
    true_vs_caption_report: dict[str, Any],
    *,
    minimum_winning_seeds: int = 2,
    minimum_control_gain: float = 0.0,
) -> dict[str, Any]:
    if minimum_winning_seeds < 1:
        raise ValueError("minimum_winning_seeds must be positive")
    if minimum_control_gain < 0:
        raise ValueError("minimum_control_gain must be nonnegative")
    random_seeds = _validate_report_header(true_vs_random_report, gate="M")
    caption_seeds = _validate_report_header(true_vs_caption_report, gate="M")
    if random_seeds != caption_seeds:
        raise GateContractError("Gate M control reports must use identical paired seeds")
    if _method_run_identity(true_vs_random_report) != _method_run_identity(
        true_vs_caption_report
    ):
        raise GateContractError("Gate M control reports must evaluate the same true-support runs")
    versus_random = _per_seed_mean_r1_delta(true_vs_random_report, random_seeds)
    versus_caption = _per_seed_mean_r1_delta(true_vs_caption_report, caption_seeds)
    per_seed = [
        {
            "seed": seed,
            "true_minus_random_mean_r1_percentage_points": random_gain,
            "true_minus_caption_mean_r1_percentage_points": caption_gain,
            "wins_both_controls": (
                random_gain > minimum_control_gain and caption_gain > minimum_control_gain
            ),
        }
        for seed, random_gain, caption_gain in zip(
            random_seeds, versus_random, versus_caption, strict=True
        )
    ]
    winning_seeds = [row["seed"] for row in per_seed if row["wins_both_controls"]]
    if len(random_seeds) < minimum_winning_seeds:
        status = "insufficient_seeds"
    elif len(winning_seeds) >= minimum_winning_seeds:
        status = "passed"
    else:
        status = "no_go"
    return {
        "schema_version": 1,
        "gate": "M",
        "status": status,
        "split": "dev",
        "paired_seeds": random_seeds,
        "observed": {
            "per_seed": per_seed,
            "winning_seeds": winning_seeds,
            "winning_seed_count": len(winning_seeds),
        },
        "criteria": {
            "minimum_winning_seeds": minimum_winning_seeds,
            "minimum_control_gain_percentage_points": minimum_control_gain,
            "strictly_greater_than_threshold": True,
        },
        "test_used_for_gate": False,
    }


def _require_completed_gate(value: dict[str, Any], name: str) -> bool:
    if value.get("schema_version") != 1 or value.get("gate") != name:
        raise GateContractError(f"Gate F requires a schema-v1 Gate {name} artifact")
    if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
        raise GateContractError(f"Gate {name} artifact violates dev-only isolation")
    status = value.get("status")
    if status not in {"passed", "no_go", "insufficient_seeds"}:
        raise GateContractError(f"Gate {name} artifact has an unknown status")
    return status == "passed"


def evaluate_full_gate(
    gain_gate: dict[str, Any],
    mechanism_gate: dict[str, Any],
    config: dict[str, Any],
    cache_meta: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    cache_artifacts_match: bool,
    train_manifest_sha256: str,
) -> dict[str, Any]:
    gain_pass = _require_completed_gate(gain_gate, "G")
    mechanism_pass = _require_completed_gate(mechanism_gate, "M")
    evidence = config.get("evidence", {})
    if not evidence.get("enabled"):
        raise GateContractError("Gate F requires evidence.enabled=true")
    if not evidence.get("require_verified_rf_metadata"):
        raise GateContractError("Gate F requires verified RF metadata")
    if not evidence.get("control_same_token_count"):
        raise GateContractError("Gate F requires token-count-matched controls")
    if cache_meta.get("schema_version") != 1 or cache_meta.get("split") != "train":
        raise GateContractError("Gate F requires a schema-v1 train-only cache")

    config_match = cache_meta.get("config_hash") == config_hash(config)
    teacher_match = cache_meta.get("teacher_hash") == config.get("model", {}).get(
        "teacher_checkpoint_sha256"
    )
    manifest_match = cache_meta.get("manifest_hash") == train_manifest_sha256

    eligible = [record for record in records if record.get("evidence_eligible") is True]
    per_video: dict[str, int] = {}
    structure_pass = True
    coordinate_pass = True
    margin_pass = True
    duration_pass = True
    tolerance = float(evidence["control_duration_tolerance"])
    threshold = float(evidence["teacher_margin_min"])
    for record in eligible:
        evidence_ids = record.get("evidence_remove_dense_indices")
        control_ids = record.get("control_remove_dense_indices")
        invalid_ids = (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or not isinstance(control_ids, list)
            or not all(isinstance(value, int) for value in evidence_ids)
            or not all(isinstance(value, int) for value in control_ids)
        )
        if invalid_ids or len(control_ids) != len(evidence_ids):
            structure_pass = False
        elif set(evidence_ids) & set(control_ids):
            structure_pass = False
        if record.get("intervention_coordinate_system") != "input_frame":
            coordinate_pass = False
        try:
            margin_pass &= float(record["teacher_clean_margin"]) >= threshold
            evidence_interval = [float(value) for value in record["evidence_interval"]]
            control_interval = [float(value) for value in record["control_interval"]]
            if len(evidence_interval) != 2 or len(control_interval) != 2:
                raise ValueError("intervention intervals must contain start/end")
            evidence_duration = evidence_interval[1] - evidence_interval[0]
            control_duration = control_interval[1] - control_interval[0]
            duration_pass &= (
                evidence_duration > 0
                and control_duration > 0
                and abs(control_duration - evidence_duration) / evidence_duration <= tolerance
            )
        except (KeyError, TypeError, ValueError, IndexError):
            structure_pass = False
            duration_pass = False
            margin_pass = False
        video_id = str(record.get("video_id"))
        per_video[video_id] = per_video.get(video_id, 0) + 1

    declared_eligible = int(cache_meta.get("gates", {}).get("evidence_eligible", -1))
    eligible_count_pass = bool(eligible) and declared_eligible == len(eligible)
    video_cap = int(evidence.get("max_pairs_per_video", 1))
    per_video_cap_pass = all(count <= video_cap for count in per_video.values())
    criteria = {
        "gain_gate_pass": gain_pass,
        "mechanism_gate_pass": mechanism_pass,
        "cache_config_hash_match": config_match,
        "cache_teacher_hash_match": teacher_match,
        "cache_manifest_hash_match": manifest_match,
        "cache_artifact_hashes_match": cache_artifacts_match,
        "evidence_count_matches_cache_gate": eligible_count_pass,
        "intervention_structure_pass": structure_pass,
        "input_frame_coordinate_pass": coordinate_pass,
        "teacher_margin_pass": margin_pass,
        "duration_matched_control_pass": duration_pass,
        "per_video_cap_pass": per_video_cap_pass,
    }
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "gate": "F",
        "status": "passed" if passed else "no_go",
        "split": "dev",
        "criteria": criteria,
        "observed": {
            "eligible_evidence_records": len(eligible),
            "declared_eligible_evidence_records": declared_eligible,
            "maximum_evidence_records_per_video": max(per_video.values(), default=0),
            "teacher_margin_min": threshold,
            "control_duration_tolerance": tolerance,
        },
        "test_used_for_gate": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate registered ELSC dev-only experiment gates"
    )
    parser.add_argument("--gate", choices=("G", "M", "F"), default="G")
    parser.add_argument("--report")
    parser.add_argument("--true-vs-random-report")
    parser.add_argument("--true-vs-caption-report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mean-r1-gain-min", type=float, default=0.5)
    parser.add_argument("--max-direction-r1-drop", type=float, default=0.5)
    parser.add_argument("--minimum-winning-seeds", type=int, default=2)
    parser.add_argument("--minimum-control-gain", type=float, default=0.0)
    parser.add_argument("--gain-gate")
    parser.add_argument("--mechanism-gate")
    parser.add_argument("--config")
    parser.add_argument("--cache")
    args = parser.parse_args(argv)
    if args.gate == "G":
        if args.report is None:
            parser.error("Gate G requires --report")
        if args.mean_r1_gain_min < 0 or args.max_direction_r1_drop < 0:
            raise ValueError("gate thresholds must be nonnegative")
        report_path = Path(args.report)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = evaluate_gain_gate(
            report,
            mean_r1_gain_min=args.mean_r1_gain_min,
            max_direction_r1_drop=args.max_direction_r1_drop,
        )
        result["comparison_report"] = str(report_path.resolve())
        result["comparison_report_sha256"] = sha256_file(report_path)
    elif args.gate == "M":
        if args.true_vs_random_report is None or args.true_vs_caption_report is None:
            parser.error(
                "Gate M requires --true-vs-random-report and --true-vs-caption-report"
            )
        random_path = Path(args.true_vs_random_report)
        caption_path = Path(args.true_vs_caption_report)
        result = evaluate_mechanism_gate(
            json.loads(random_path.read_text(encoding="utf-8")),
            json.loads(caption_path.read_text(encoding="utf-8")),
            minimum_winning_seeds=args.minimum_winning_seeds,
            minimum_control_gain=args.minimum_control_gain,
        )
        result["comparison_reports"] = {
            "true_vs_random": str(random_path.resolve()),
            "true_vs_random_sha256": sha256_file(random_path),
            "true_vs_caption": str(caption_path.resolve()),
            "true_vs_caption_sha256": sha256_file(caption_path),
        }
    else:
        required = {
            "--gain-gate": args.gain_gate,
            "--mechanism-gate": args.mechanism_gate,
            "--config": args.config,
            "--cache": args.cache,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("Gate F requires " + ", ".join(missing))
        gain_path = Path(args.gain_gate)
        mechanism_path = Path(args.mechanism_gate)
        config_path = Path(args.config)
        cache_path = Path(args.cache)
        meta_path = cache_path / "cache_meta.json"
        records_path = cache_path / "records.jsonl"
        cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
        ]
        config = load_config(config_path, stage="train")
        train_manifest_sha256 = sha256_file(config["data"]["train_manifest"])
        artifact_paths = {
            "records_sha256": records_path,
            "negative_table_hash": cache_path / "negative_graph.json",
            "lexical_bank_hash": cache_path / "lexical_bank.npy",
        }
        artifact_integrity = {
            key: path.is_file() and cache_meta.get(key) == sha256_file(path)
            for key, path in artifact_paths.items()
        }
        result = evaluate_full_gate(
            json.loads(gain_path.read_text(encoding="utf-8")),
            json.loads(mechanism_path.read_text(encoding="utf-8")),
            config,
            cache_meta,
            records,
            cache_artifacts_match=all(artifact_integrity.values()),
            train_manifest_sha256=train_manifest_sha256,
        )
        result["inputs"] = {
            "gain_gate": str(gain_path.resolve()),
            "gain_gate_sha256": sha256_file(gain_path),
            "mechanism_gate": str(mechanism_path.resolve()),
            "mechanism_gate_sha256": sha256_file(mechanism_path),
            "config": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "cache_meta": str(meta_path.resolve()),
            "cache_meta_sha256": sha256_file(meta_path),
            "cache_artifact_integrity": artifact_integrity,
        }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
