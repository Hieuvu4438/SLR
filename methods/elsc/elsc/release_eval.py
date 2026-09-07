from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from elsc.config import config_hash, load_config
from elsc.evaluate import evaluate_model
from slr_common.resources import require_resources
from elsc.upstream.factory import build_retriever_from_checkpoint
from slr_common.utils import atomic_json_dump, sha256_file


PHOENIX_PUBLISHED = {
    "T2V": {"R1": 69.5, "R5": 86.6, "R10": 92.1, "MedianR": 1.0},
    "V2T": {"R1": 70.2, "R5": 88.0, "R10": 92.8, "MedianR": 1.0},
}


def evaluate_release(
    config_path: Path, output: Path, device: torch.device
) -> dict[str, Any]:
    config = load_config(config_path, stage="evaluate")
    if config.get("method") != "baseline" or config["model"]["adapter"].get("enabled"):
        raise ValueError("release evaluation requires the unmodified baseline architecture")
    manifest = Path(config["data"]["test_manifest"])
    checkpoint = Path(config["model"]["init_checkpoint"])
    release_provenance = Path("artifacts/sign_features/release_provenance.json")
    for path in (manifest, checkpoint, release_provenance):
        if not path.is_file():
            raise FileNotFoundError(f"release evaluation asset is missing: {path}")
    resources = config.get("resources", {})
    resource_snapshot = require_resources(
        output,
        device,
        min_disk_gib=float(resources.get("min_free_disk_gib", 20)),
        min_gpu_gib=float(resources.get("min_free_gpu_gib_evaluation", 4)),
        operation="official release test parity evaluation",
    )
    model, _ = build_retriever_from_checkpoint(config, checkpoint, device=device)
    scores, metrics = evaluate_model(model, config, "test", device)
    deltas: dict[str, dict[str, float]] = {}
    for direction, expected in PHOENIX_PUBLISHED.items():
        deltas[direction] = {
            metric: float(metrics[direction][metric]) - value for metric, value in expected.items()
        }
    passed = all(
        abs(delta) <= 0.11
        for direction in deltas.values()
        for metric, delta in direction.items()
        if metric != "MedianR"
    ) and all(
        abs(direction["MedianR"]) <= 1e-9 for direction in deltas.values()
    )
    metrics.update(
        {
            "status": "passed" if passed else "metric_mismatch",
            "result_kind": "official_release_artifact_parity_only",
            "selection_eligible": False,
            "test_used_for_training_selection": False,
            "published_reference": PHOENIX_PUBLISHED,
            "delta_from_published_percentage_points": deltas,
            "comparison_tolerance_percentage_points": 0.11,
            "config": str(config_path.resolve()),
            "config_hash": config_hash(config),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
                "role": "official_release_checkpoint",
            },
            "test_manifest": {
                "path": str(manifest.resolve()),
                "sha256": sha256_file(manifest),
            },
            "feature_release_provenance": {
                "path": str(release_provenance.resolve()),
                "sha256": sha256_file(release_provenance),
            },
            "resources": resource_snapshot,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "scores_video_x_text.npy", scores)
    atomic_json_dump(metrics, output / "metrics.json")
    atomic_json_dump(
        {
            "schema_version": 1,
            "result_kind": "official_release_artifact_parity_only",
            "selection_eligible": False,
            "split": "test",
            "score_orientation": "video_x_text",
            "V2T": metrics["per_query"]["V2T"],
            "T2V": metrics["per_query"]["T2V"],
        },
        output / "per_query_ranks.json",
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the untouched CiCo release checkpoint on official release features"
    )
    parser.add_argument("--config", default="configs/ph_release.yaml")
    parser.add_argument("--output", default="artifacts/release_eval/ph_test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = evaluate_release(Path(args.config), Path(args.output), torch.device(args.device))
    print(
        json.dumps(
            {
                "status": result["status"],
                "T2V": result["T2V"],
                "V2T": result["V2T"],
                "delta": result["delta_from_published_percentage_points"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
