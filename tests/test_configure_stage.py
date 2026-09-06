from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

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
