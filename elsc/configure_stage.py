from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from elsc.config import dump_resolved, load_config, validate_config
from elsc.provenance import validate_dev_selection
from elsc.utils import atomic_json_dump, git_worktree_state, sha256_file


def _selected_run(run: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    checkpoint = run / "checkpoints" / "best_dev.pt"
    selection_path = run / "selection.json"
    selection = validate_dev_selection(selection_path, checkpoint)
    config_path = run / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"selected run config is missing: {config_path}")
    return checkpoint, selection_path, selection, load_config(config_path, validate=False)


def configure_from_teacher(
    template_path: Path,
    teacher_run: Path,
    output: Path,
    *,
    student_run: Path | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    template = load_config(template_path, validate=False)
    teacher_checkpoint, selection_path, selection, teacher_config = _selected_run(
        teacher_run
    )
    initialization_run = student_run or teacher_run
    (
        initialization_checkpoint,
        initialization_selection_path,
        initialization_selection,
        initialization_config,
    ) = _selected_run(initialization_run)
    teacher_seed = int(teacher_config["seed"])
    initialization_seed = int(initialization_config["seed"])
    if initialization_seed != teacher_seed:
        raise ValueError(
            "teacher and student-initialization runs must use the same seed: "
            f"{teacher_seed} != {initialization_seed}"
        )
    teacher_sha = sha256_file(teacher_checkpoint)
    initialization_sha = sha256_file(initialization_checkpoint)
    template["seed"] = teacher_seed
    template["experiment"] = f"{template['experiment']}_s{template['seed']}"
    model = template["model"]
    model.update(
        {
            "init_checkpoint": str(initialization_checkpoint),
            "init_checkpoint_sha256": initialization_sha,
            "init_checkpoint_role": (
                "same_dev_selected_baseline_as_teacher"
                if student_run is None
                else "dev_selected_student_stage"
            ),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": teacher_sha,
            "teacher_selection_provenance": str(selection_path),
        }
    )
    if cache_path is not None:
        template.setdefault("cache", {})["path"] = str(cache_path)
    if model.get("require_init_equals_teacher", False) and initialization_sha != teacher_sha:
        raise ValueError("template requires student initialization to equal teacher checkpoint")
    validate_config(template, stage="train")
    digest = dump_resolved(template, output)
    record = {
        "schema_version": 1,
        "implementation": git_worktree_state(Path(__file__)),
        "template": str(template_path.resolve()),
        "template_sha256": sha256_file(template_path),
        "teacher_run": str(teacher_run.resolve()),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_selection": str(selection_path.resolve()),
        "teacher_selection_sha256": sha256_file(selection_path),
        "teacher_selected_epoch": selection.get("selected_epoch"),
        "student_initialization_run": str(initialization_run.resolve()),
        "student_initialization_checkpoint": str(initialization_checkpoint.resolve()),
        "student_initialization_checkpoint_sha256": initialization_sha,
        "student_initialization_selection": str(initialization_selection_path.resolve()),
        "student_initialization_selection_sha256": sha256_file(
            initialization_selection_path
        ),
        "student_initialization_selected_epoch": initialization_selection.get(
            "selected_epoch"
        ),
        "cache_path": template.get("cache", {}).get("path"),
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
    parser.add_argument(
        "--student-run",
        help="Optional dev-selected prior stage used only for student initialization",
    )
    parser.add_argument(
        "--cache-path",
        help="Optional run-specific train-only cache path (recommended across seeds)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = configure_from_teacher(
        Path(args.template),
        Path(args.teacher_run),
        Path(args.output),
        student_run=Path(args.student_run) if args.student_run else None,
        cache_path=Path(args.cache_path) if args.cache_path else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
