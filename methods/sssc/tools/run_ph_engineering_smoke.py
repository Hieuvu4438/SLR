from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(METHOD_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "shared"))

from method1.config import load_config  # noqa: E402
from method1.train_pipeline import train_stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a non-reportable PH engineering smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()
    config = load_config(METHOD_ROOT / "configs" / "method1" / "ph_base_initial.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            global_contrastive_batch=2,
            num_workers=0,
            checkpoint_deterministic_score_blocks=True,
        ),
        evaluation=replace(config.evaluation, encode_batch_size=16),
        output=replace(
            config.output,
            root=str(
                REPOSITORY_ROOT
                / "runs"
                / "method1"
                / "ph"
                / "base_engineering_smoke"
                / "seed42"
            ),
            save_last=False,
            save_best_dev=False,
        ),
    )
    report = train_stage(
        config,
        stage="base",
        max_steps=args.max_steps,
        requested_device=args.device,
        upret_root=REPOSITORY_ROOT / "third_party" / "UPRet",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
