from __future__ import annotations

from pathlib import Path

from elsc.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_csl_full_and_continued_min_templates_are_step_matched():
    full = load_config(ROOT / "configs/csl_full.yaml", validate=False)
    continued = load_config(ROOT / "configs/csl_continued_min.yaml", validate=False)

    assert full["data"]["dataset"] == "csl_daily"
    assert full["sources"]["prepared_splits"] == ["train", "dev"]
    assert full["sources"]["test_annotation"] is None
    assert full["train"]["one_video_per_caption_group"] is True
    assert full["train"]["epochs"] == continued["train"]["epochs"] == 10

    matched_train_fields = {
        "per_device_batch",
        "accumulation_steps",
        "optimizer",
        "beta1",
        "beta2",
        "epsilon",
        "exclude_bias_and_1d_from_weight_decay",
        "core_lr",
        "adapter_lr",
        "head_lr",
        "weight_decay",
        "warmup_ratio",
        "schedule",
        "grad_clip_norm",
        "precision",
        "checkpoint_metric",
    }
    for field in matched_train_fields:
        assert full["train"][field] == continued["train"][field]


def test_csl_full_template_enforces_rf_matched_evidence_controls():
    full = load_config(ROOT / "configs/csl_full.yaml", validate=False)
    evidence = full["evidence"]

    assert evidence["enabled"] is True
    assert evidence["require_verified_rf_metadata"] is True
    assert evidence["control_same_token_count"] is True
    assert evidence["control_duration_tolerance"] == 0.10
    assert full["sources"]["require_temporal_metadata"] is True
    assert full["sources"]["temporal_metadata_root"].endswith("csl_temporal_metadata")
    assert full["model"]["require_init_equals_teacher"] is False
