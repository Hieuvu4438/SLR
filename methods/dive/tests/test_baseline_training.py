from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from dive.adapters import SedsTextBatch, SedsTrainingBatch, SedsVideoBatch, load_seds_reproduction
from dive.baseline_training import (
    build_baseline_epoch_plan,
    build_seds_optimizer,
    run_seds_training_step,
)
from dive.data.manifest import SampleRecord


ROOT = Path(__file__).resolve().parents[3]
SEDS_ROOT = ROOT / "third_party" / "SEDS"
REPRODUCTION_CONFIG = ROOT / "methods" / "dive" / "configs" / "seds_how2sign_reproduction.yaml"


def _record(sample_id: str, text_id: str) -> SampleRecord:
    return SampleRecord(
        schema_version="sample.v1",
        sample_id=sample_id,
        video_id=sample_id,
        text_id=text_id,
        split="train",
        sign_language="ase",
        text_language_original="en",
        text_language_model="en",
        text_original=f"caption {text_id}",
        text_model=f"caption {text_id}",
        source_video_id="source",
        signer_id=None,
        source_start_sec=0.0,
        source_end_sec=1.0,
        duration_sec=1.0,
        video_path=f"train/{sample_id}.mp4",
        pose_path=f"train/{sample_id}.pkl",
        rgb_feature_key=f"train/{sample_id}.pkl",
        translation_artifact_hash=None,
        frame_map_key=sample_id,
        annotation_provenance="fixture",
    )


def test_epoch_plan_is_deterministic_and_selects_one_unique_view_per_text():
    records = [_record("v0", "t0"), _record("v1", "t0"), _record("v2", "t1")]
    first = build_baseline_epoch_plan(records, seed=17, epoch=3)
    second = build_baseline_epoch_plan(tuple(reversed(records)), seed=17, epoch=3)
    assert first.ordered_sample_ids == second.ordered_sample_ids
    assert first.sha256 == second.sha256
    assert set(first.ordered_text_ids) == {"t0", "t1"}
    assert len(first.ordered_text_ids) == len(set(first.ordered_text_ids)) == 2
    assert (
        first.augmentation_seed
        != build_baseline_epoch_plan(records, seed=17, epoch=4).augmentation_seed
    )


class _GroupedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip = nn.Linear(2, 2)
        self.clip_rgb = nn.Linear(2, 2)
        self.signbert = nn.Linear(2, 2)
        self.other = nn.Linear(2, 2)


def test_optimizer_reproduces_six_native_parameter_groups():
    reproduction = load_seds_reproduction(REPRODUCTION_CONFIG, upstream_root=SEDS_ROOT)
    model = _GroupedModel()
    optimizer, manifest = build_seds_optimizer(
        model,
        upstream_root=SEDS_ROOT,
        controlled_training=reproduction.controlled_training,
        total_steps=11,
    )
    groups = manifest["groups"]
    assert [(item["decay"], item["branch"]) for item in groups] == [
        (True, "clip"),
        (True, "signbert"),
        (True, "other"),
        (False, "clip"),
        (False, "signbert"),
        (False, "other"),
    ]
    assert [item["lr"] for item in groups] == [1e-5, 1e-4, 1e-4, 1e-5, 1e-4, 1e-4]
    names = [name for item in groups for name in item["parameter_names"]]
    assert sorted(names) == sorted(name for name, _ in model.named_parameters())
    assert len(optimizer.param_groups) == 6


class _TrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(math.log(1000.0)))

    def forward(self, input_ids, token_type_ids, attention_mask, right, left, body, aug, aug_mask):
        del token_type_ids, attention_mask, right, left, body, aug, aug_mask
        loss = self.weight.square() + input_ids.float().sum() * 0
        return loss, loss / 2, loss / 3, loss / 4, 0.0, 0.0, loss / 5


def _training_batch() -> SedsTrainingBatch:
    text = SedsTextBatch(
        text_ids=("t0",),
        input_ids=torch.ones(1, 3, dtype=torch.long),
        token_type_ids=torch.zeros(1, 3, dtype=torch.long),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )
    video = SedsVideoBatch(
        sample_ids=("v0",),
        right_pose=torch.zeros(1, 16, 21, 2),
        left_pose=torch.zeros(1, 16, 21, 2),
        body_pose=torch.zeros(1, 16, 7, 2),
        clip_starts=torch.zeros(1, 1, dtype=torch.long),
        legacy_video_mask=torch.zeros(1, 2, dtype=torch.long),
        rgb_features=torch.zeros(1, 1024, 1, 1),
        grid_id="fixture",
        raw_frame_counts=(16,),
        pose_raw_frame_indices=(tuple(range(16)),),
    )
    return SedsTrainingBatch(
        video=video,
        text=text,
        augmented_text=text,
        augmented_strings=("caption",),
        augmented=(False,),
    )


def test_training_step_uses_native_objective_clips_gradients_and_logit_scale():
    model = _TrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = run_seds_training_step(
        model,
        optimizer,
        _training_batch(),
        device="cpu",
        global_grad_clip_norm=1.0,
    )
    assert metrics.loss == 4.0
    assert metrics.global_grad_norm == 4.0
    assert model.weight.item() < 2.0
    assert model.clip.logit_scale.exp().item() <= 100.0001
