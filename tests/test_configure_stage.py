from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
import pytest

from elsc.configure_stage import configure_from_teacher
from elsc.utils import sha256_file


def test_auxiliary_config_is_bound_to_dev_selected_teacher(tmp_path: Path):
    run = tmp_path / "baseline"
    checkpoint = run / "checkpoints" / "best_dev.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint)
    (run / "resolved_config.yaml").write_text("seed: 42\n", encoding="utf-8")
    selection = {
        "selection_split": "dev",
        "test_used_for_selection": False,
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_epoch": 3,
    }
    (run / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    template = tmp_path / "template.yaml"
    template.write_text(
        """
schema_version: 1
experiment: min
method: elsc
sources: {}
data: {caption_language: en}
model:
  backbone_frozen: true
  adapter: {enabled: true, radius: 0}
cache: {}
train: {eval_split: dev, epochs: 1}
evaluation: {filter_by_aux_eligibility: false}
""",
        encoding="utf-8",
    )
    output = tmp_path / "resolved.yaml"
    result = configure_from_teacher(template, run, output)
    resolved = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert resolved["model"]["teacher_checkpoint"] == str(checkpoint)
    assert resolved["model"]["init_checkpoint_sha256"] == sha256_file(checkpoint)
    assert resolved["seed"] == 42
    assert result["teacher_selected_epoch"] == 3


def test_full_config_uses_selected_min_student_and_baseline_teacher(tmp_path: Path):
    def selected_run(name: str, weight: float) -> tuple[Path, Path]:
        run = tmp_path / name
        checkpoint = run / "checkpoints" / "best_dev.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"model": {"weight": torch.tensor([weight])}}, checkpoint)
        (run / "resolved_config.yaml").write_text("seed: 42\n", encoding="utf-8")
        (run / "selection.json").write_text(
            json.dumps(
                {
                    "selection_split": "dev",
                    "test_used_for_selection": False,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "selected_epoch": 2,
                }
            ),
            encoding="utf-8",
        )
        return run, checkpoint

    baseline_run, baseline_checkpoint = selected_run("baseline", 1.0)
    min_run, min_checkpoint = selected_run("min", 2.0)
    template = tmp_path / "full.yaml"
    template.write_text(
        """
schema_version: 1
experiment: full
method: elsc
sources: {temporal_metadata_root: /rf}
data: {caption_language: en}
model:
  backbone_frozen: true
  require_init_equals_teacher: false
  adapter: {enabled: true, radius: 0}
cache: {}
evidence: {enabled: true, control_same_token_count: true, mask_fill: zero}
train: {eval_split: dev, epochs: 1}
evaluation: {filter_by_aux_eligibility: false}
""",
        encoding="utf-8",
    )
    output = tmp_path / "full_resolved.yaml"
    result = configure_from_teacher(
        template,
        baseline_run,
        output,
        student_run=min_run,
        cache_path=tmp_path / "cache_s42",
    )
    resolved = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert resolved["model"]["teacher_checkpoint"] == str(baseline_checkpoint)
    assert resolved["model"]["init_checkpoint"] == str(min_checkpoint)
    assert resolved["cache"]["path"] == str(tmp_path / "cache_s42")
    assert result["cache_path"] == str(tmp_path / "cache_s42")
    assert result["teacher_checkpoint_sha256"] == sha256_file(baseline_checkpoint)
    assert result["student_initialization_checkpoint_sha256"] == sha256_file(
        min_checkpoint
    )


def test_student_initialization_requires_same_seed(tmp_path: Path):
    def selected_run(name: str, seed: int) -> Path:
        run = tmp_path / name
        checkpoint = run / "checkpoints" / "best_dev.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"model": {}}, checkpoint)
        (run / "resolved_config.yaml").write_text(
            f"seed: {seed}\n", encoding="utf-8"
        )
        (run / "selection.json").write_text(
            json.dumps(
                {
                    "selection_split": "dev",
                    "test_used_for_selection": False,
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            ),
            encoding="utf-8",
        )
        return run

    baseline_run = selected_run("baseline", 42)
    min_run = selected_run("min", 7)
    template = tmp_path / "full.yaml"
    template.write_text(
        """
schema_version: 1
experiment: full
method: elsc
sources: {temporal_metadata_root: /rf}
data: {caption_language: en}
model:
  backbone_frozen: true
  require_init_equals_teacher: false
  adapter: {enabled: true, radius: 0}
cache: {}
evidence: {enabled: true, control_same_token_count: true, mask_fill: zero}
train: {eval_split: dev, epochs: 1}
evaluation: {filter_by_aux_eligibility: false}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="same seed"):
        configure_from_teacher(
            template,
            baseline_run,
            tmp_path / "resolved.yaml",
            student_run=min_run,
        )
