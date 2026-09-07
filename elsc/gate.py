from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elsc.utils import atomic_json_dump, sha256_file


class GateContractError(ValueError):
    pass


def evaluate_gain_gate(
    report: dict[str, Any],
    *,
    mean_r1_gain_min: float = 0.5,
    max_direction_r1_drop: float = 0.5,
) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise GateContractError("comparison report must use schema_version=1")
    if report.get("result_kind") != "measured_local":
        raise GateContractError("Gate G requires measured_local results")
    if report.get("split") != "dev":
        raise GateContractError("Gate G may only use the dev split")
    if report.get("units") != "percentage_points":
        raise GateContractError("Gate G requires percentage-point deltas")
    paired_seeds = report.get("paired_seeds")
    if not isinstance(paired_seeds, list) or not paired_seeds:
        raise GateContractError("Gate G requires at least one paired seed")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the registered ELSC practical dev-gain gate"
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mean-r1-gain-min", type=float, default=0.5)
    parser.add_argument("--max-direction-r1-drop", type=float, default=0.5)
    args = parser.parse_args(argv)
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
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
