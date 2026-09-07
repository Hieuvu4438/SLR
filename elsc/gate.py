from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate registered ELSC dev-only experiment gates"
    )
    parser.add_argument("--gate", choices=("G", "M"), default="G")
    parser.add_argument("--report")
    parser.add_argument("--true-vs-random-report")
    parser.add_argument("--true-vs-caption-report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mean-r1-gain-min", type=float, default=0.5)
    parser.add_argument("--max-direction-r1-drop", type=float, default=0.5)
    parser.add_argument("--minimum-winning-seeds", type=int, default=2)
    parser.add_argument("--minimum-control-gain", type=float, default=0.0)
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
    else:
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
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
