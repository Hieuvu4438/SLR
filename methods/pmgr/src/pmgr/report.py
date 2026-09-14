from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slr_common.utils import atomic_json_dump, sha256_file


ARM_CONTRACTS = {
    "C0": "legacy_cico",
    "C1": "single_mixed_ce",
    "C2": "all_uniform_ce",
    "C3": "all_set_ce",
    "C23_population": (
        "all_uniform_ce_population_weighted",
        "all_set_ce_population_weighted",
    ),
    "C4": "group_ce",
}


def _read_run(path: Path, expected_mode: str | tuple[str, ...]) -> dict[str, Any]:
    summary_path = path / "summary.json"
    config_path = path / "resolved_config.json"
    metrics_path = path / "best_dev_metrics.json"
    missing = [str(item) for item in (summary_path, config_path, metrics_path) if not item.is_file()]
    if missing:
        raise ValueError("run is incomplete; missing " + ", ".join(missing))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    best = json.loads(metrics_path.read_text(encoding="utf-8"))
    if summary.get("status") != "training_complete":
        raise ValueError(f"run did not complete training: {path}")
    actual_mode = config.get("loss", {}).get("mode")
    allowed_modes = (expected_mode,) if isinstance(expected_mode, str) else expected_mode
    if actual_mode not in allowed_modes:
        raise ValueError(f"run {path} is not one of the expected {allowed_modes} arms")
    if best.get("metrics", {}).get("split") != "validation":
        raise ValueError(f"run {path} lacks validation-selected metrics")
    metrics = best["metrics"]
    primary = 0.5 * (metrics["T2V"]["R1"] + metrics["V2T"]["R1"])
    log_path = path / "train.jsonl"
    exposures = {
        "loaded_videos": 0,
        "candidate_pairs": 0,
        "encoded_video_items": 0,
        "encoded_text_views": 0,
        "effective_steps": 0,
        "accelerator_seconds": 0.0,
        "seconds_per_update": 0.0,
        "peak_allocated_gpu_bytes": 0,
    }
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if "loaded_videos" not in value:
                continue
            exposures["loaded_videos"] += int(value["loaded_videos"])
            exposures["candidate_pairs"] += int(value["candidate_pairs"])
            exposures["encoded_video_items"] += int(value["encoded_video_items"])
            exposures["encoded_text_views"] += int(value["encoded_text_views"])
            exposures["effective_steps"] += 1
            exposures["accelerator_seconds"] += float(value["accelerator_seconds"])
            exposures["seconds_per_update"] += float(value["seconds_per_update"])
            exposures["peak_allocated_gpu_bytes"] = max(
                exposures["peak_allocated_gpu_bytes"],
                int(value["peak_allocated_gpu_bytes"]),
            )
    return {
        "path": str(path),
        "loss_mode": actual_mode,
        "checkpoint": summary["checkpoint"],
        "best_epoch": best["epoch"],
        "primary_mean_bidirectional_r1": primary,
        "T2V": {name: metrics["T2V"][name] for name in ("R1", "R5", "R10", "MedianR", "MeanR")},
        "V2T": {name: metrics["V2T"][name] for name in ("R1", "R5", "R10", "MedianR", "MeanR")},
        "exposures": exposures,
        "resolved_config_sha256": sha256_file(config_path),
    }


def build_phase_b_report(runs: dict[str, Path]) -> dict[str, Any]:
    if set(runs) != set(ARM_CONTRACTS):
        raise ValueError(f"phase-B report requires arms {sorted(ARM_CONTRACTS)}")
    arms = {name: _read_run(runs[name], mode) for name, mode in ARM_CONTRACTS.items()}
    strongest_positive = max(
        ("C2", "C3"),
        key=lambda name: arms[name]["primary_mean_bidirectional_r1"],
    )
    expected_population_mode = {
        "C2": "all_uniform_ce_population_weighted",
        "C3": "all_set_ce_population_weighted",
    }[strongest_positive]
    if arms["C23_population"]["loss_mode"] != expected_population_mode:
        raise ValueError(
            "population-weighted arm does not match the stronger C2/C3 positive control"
        )
    equal_input_controls = ("C2", "C3", "C23_population")
    strongest = max(
        equal_input_controls,
        key=lambda name: arms[name]["primary_mean_bidirectional_r1"],
    )
    delta = (
        arms["C4"]["primary_mean_bidirectional_r1"]
        - arms[strongest]["primary_mean_bidirectional_r1"]
    )
    return {
        "schema_version": 1,
        "phase": "B_population_only",
        "split": "validation",
        "test_accessed": False,
        "arms": arms,
        "strongest_ordinary_positive_control": strongest_positive,
        "selected_population_weighted_mode": expected_population_mode,
        "strongest_equal_input_positive_control": strongest,
        "c4_delta_mean_bidirectional_r1_points": delta,
        "planning_threshold_points": 0.5,
        "status": (
            "population_signal_requires_mechanism_audit_and_replication"
            if delta >= 0.5
            else "population_hypothesis_no_go"
        ),
        "research_supported": False,
        "reason": "Phase B contains one seed; rank extension and cross-dataset replication are not run",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a measured PMGR Phase-B report")
    for arm in ARM_CONTRACTS:
        parser.add_argument(f"--{arm.lower().replace('_', '-')}", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    runs = {
        arm: Path(getattr(args, arm.lower()))
        for arm in ARM_CONTRACTS
    }
    result = build_phase_b_report(runs)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
