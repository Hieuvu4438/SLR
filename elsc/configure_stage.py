from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elsc.config import dump_resolved, load_config, validate_config
from elsc.provenance import validate_dev_selection
from elsc.utils import atomic_json_dump, sha256_file


def configure_from_teacher(
    template_path: Path, teacher_run: Path, output: Path
) -> dict[str, Any]:
    template = load_config(template_path, validate=False)
    teacher_checkpoint = teacher_run / "checkpoints" / "best_dev.pt"
    selection_path = teacher_run / "selection.json"
    selection = validate_dev_selection(selection_path, teacher_checkpoint)
    teacher_config_path = teacher_run / "resolved_config.yaml"
    if not teacher_config_path.is_file():
        raise FileNotFoundError(f"teacher resolved config is missing: {teacher_config_path}")
    teacher_config = load_config(teacher_config_path, validate=False)
    teacher_sha = sha256_file(teacher_checkpoint)
    template["seed"] = int(teacher_config["seed"])
    template["experiment"] = f"{template['experiment']}_s{template['seed']}"
    model = template["model"]
    model.update(
        {
            "init_checkpoint": str(teacher_checkpoint),
            "init_checkpoint_sha256": teacher_sha,
            "init_checkpoint_role": "same_dev_selected_baseline_as_teacher",
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": teacher_sha,
            "teacher_selection_provenance": str(selection_path),
        }
    )
    validate_config(template, stage="train")
    digest = dump_resolved(template, output)
    record = {
        "schema_version": 1,
        "template": str(template_path.resolve()),
        "template_sha256": sha256_file(template_path),
        "teacher_run": str(teacher_run.resolve()),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_selection": str(selection_path.resolve()),
        "teacher_selection_sha256": sha256_file(selection_path),
        "teacher_selected_epoch": selection.get("selected_epoch"),
        "output": str(output.resolve()),
        "resolved_config_hash": digest,
    }
    atomic_json_dump(record, output.with_suffix(output.suffix + ".provenance.json"))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve an auxiliary-stage config against a dev-selected baseline teacher"
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--teacher-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = configure_from_teacher(
        Path(args.template), Path(args.teacher_run), Path(args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
