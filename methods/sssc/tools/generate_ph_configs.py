from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "method1" / "ph_base_initial.yaml"
ARMS = {
    "base_initial": ("shared", "base"),
    "base_continuation": ("shared", "base_continuation"),
    "span_independent": ("independent", "span_independent"),
    "span_shared": ("shared", "span_shared"),
    "span_random_support": ("random", "span_random_support"),
    "caption_hn": ("candidate", "caption_hn"),
    "fsc_local": ("candidate", "fsc_local"),
    "fsc_local_caption_hn": ("candidate", "fsc_local_caption_hn"),
}


def main() -> None:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    for seed in (42, 43, 44):
        for arm, (support_mode, output_name) in ARMS.items():
            value = copy.deepcopy(source)
            value["seed"] = seed
            value["reference"]["checkpoint"] = (
                f"runs/method1/ph/base/seed{seed}/best_dev.pt"
            )
            value["reference"]["cache_dir"] = (
                f"artifacts/method1/reference/ph/seed{seed}"
            )
            value["auxiliary"]["arm"] = arm
            value["auxiliary"]["support_mode"] = support_mode
            value["output"]["root"] = f"runs/method1/ph/{output_name}/seed{seed}"
            destination = ROOT / "configs" / "method1" / f"ph_seed{seed}_{arm}.yaml"
            destination.write_text(
                yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
