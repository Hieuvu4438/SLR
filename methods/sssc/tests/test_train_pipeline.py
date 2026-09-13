from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import method1.train_pipeline as pipeline
from method1.config import load_config


class TinyDistribution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(-1.0))

    def forward(self, tokens, mask=None, weight=None):
        return tokens, self.log_sigma.expand_as(tokens), tokens


class TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.video_weight_fc = nn.Linear(2, 1)
        self.text_weight_fc = nn.Linear(2, 1)
        self.dist_text_trans = TinyDistribution()
        self.dist_video_trans = TinyDistribution()
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.eps = 0.1
        self.max_iter = 10
        self.ot_weight = 1.0
        self.dual_mix = 0.5
        self.mix_design = "balance"

    def get_text_video_feat(
        self,
        input_ids,
        token_type_ids,
        text_valid,
        video_features,
        video_ignore_raw,
        **kwargs,
    ):
        batch = input_ids.shape[0]
        identity = video_features[:, 0, 0, 0].float()
        local = torch.stack((identity + 1.0, 2.0 - identity), dim=-1)
        video = local[:, None, :].expand(batch, 65, 2) * self.scale
        text_identity = input_ids[:, 0].float()
        text_base = torch.stack((text_identity + 1.0, 2.0 - text_identity), dim=-1)
        text = text_base[:, None, :].expand(batch, 2, 2) * self.scale
        augmented = (text + 0.03) * self.scale
        return (
            text,
            text_valid,
            text[:, 0],
            video,
            video_ignore_raw,
            video[:, 0],
            augmented,
            text_valid,
            augmented[:, 0],
        )


class TinyDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        video_ignore = torch.ones(65, dtype=torch.bool)
        video_ignore[1:3] = False
        features = torch.zeros(1024, 64, 1)
        features[0, 0, 0] = float(index)
        return {
            "video_uid": f"v{index}",
            "text_uid": f"t{index}",
            "group_uid": f"g{index}",
            "caption_hash": f"h{index}",
            "canonical_text": f"text {index}",
            "lexical_spans": (),
            "video_features": features,
            "video_ignore_raw": video_ignore,
            "selected_feature_indices": torch.tensor([0, 1, *([-1] * 62)]),
            "input_ids": torch.tensor([index, 2]),
            "text_valid": torch.tensor([True, True]),
            "input_ids_aug": torch.tensor([index, 2]),
            "text_aug_valid": torch.tensor([True, True]),
            "token_type_ids": torch.zeros(2, dtype=torch.long),
        }


class FourItemDataset(TinyDataset):
    def __len__(self) -> int:
        return 4


class InterruptAfterFirstBatchDataset(FourItemDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests = 0

    def __getitem__(self, index: int):
        self.requests += 1
        if self.requests > 2:
            raise RuntimeError("simulated interruption after periodic checkpoint")
        return super().__getitem__(index)


def test_one_step_training_pipeline_writes_completed_selected_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    base = load_config("methods/sssc/configs/method1/ph_base_initial.yaml")
    config = replace(
        base,
        training=replace(
            base.training,
            base_epochs=1,
            global_contrastive_batch=2,
            num_workers=0,
            checkpoint_deterministic_score_blocks=False,
        ),
        output=replace(base.output, root=str(tmp_path / "run")),
    )
    student = TinyStudent()
    monkeypatch.setattr(
        pipeline,
        "build_upret_model",
        lambda *args, **kwargs: (
            student,
            {"trainable_count": 10, "total_count": 10},
        ),
    )
    monkeypatch.setattr(pipeline, "create_upret_tokenizer", lambda *args: object())
    monkeypatch.setattr(pipeline, "Method1Dataset", TinyDataset)
    monkeypatch.setattr(
        pipeline,
        "evaluate_loaded_student",
        lambda *args, **kwargs: {
            "T2V": {"R1": 10.0, "R5": 20.0},
            "V2T": {"R1": 30.0, "R5": 40.0},
        },
    )
    report = pipeline.train_stage(config, stage="base", requested_device="cpu")
    assert report["status"] == "complete"
    assert report["global_step"] == 1
    checkpoint = torch.load(
        tmp_path / "run" / "best_dev.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["training_run_complete"] is True
    assert checkpoint["arm"] == "base_initial"
    assert checkpoint["dev_selection"]["mean_bidirectional_r1"] == 20.0
    assert (tmp_path / "run" / "training_complete.json").is_file()


def test_k1_method_pilot_rejects_more_than_200_steps() -> None:
    config = load_config(
        "methods/sssc/configs/method1/ph_seed42_span_shared_k1_pilot.yaml"
    )
    with pytest.raises(pipeline.TrainingError, match="capped at 200"):
        pipeline.train_stage(config, stage="method", max_steps=201)


def test_periodic_checkpoint_resume_matches_continuous_updates(
    tmp_path: Path, monkeypatch
) -> None:
    base = load_config("methods/sssc/configs/method1/ph_base_initial.yaml")

    def configured(output: Path):
        return replace(
            base,
            training=replace(
                base.training,
                base_epochs=1,
                global_contrastive_batch=2,
                num_workers=0,
                checkpoint_deterministic_score_blocks=False,
                checkpoint_every_steps=1,
            ),
            output=replace(base.output, root=str(output)),
        )

    def build(*args, **kwargs):
        torch.manual_seed(77)
        return TinyStudent(), {"trainable_count": 10, "total_count": 10}

    monkeypatch.setattr(pipeline, "build_upret_model", build)
    monkeypatch.setattr(pipeline, "create_upret_tokenizer", lambda *args: object())
    monkeypatch.setattr(
        pipeline,
        "evaluate_loaded_student",
        lambda *args, **kwargs: {
            "T2V": {"R1": 10.0, "R5": 20.0},
            "V2T": {"R1": 30.0, "R5": 40.0},
        },
    )

    interrupted_config = configured(tmp_path / "interrupted")
    monkeypatch.setattr(pipeline, "Method1Dataset", InterruptAfterFirstBatchDataset)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.train_stage(interrupted_config, stage="base", requested_device="cpu")
    periodic_path = tmp_path / "interrupted" / "last.pt"
    periodic = torch.load(periodic_path, map_location="cpu", weights_only=True)
    assert periodic["global_step"] == 1
    assert periodic["next_batch_index"] == 1
    assert periodic["training_run_complete"] is False

    monkeypatch.setattr(pipeline, "Method1Dataset", FourItemDataset)
    resumed = pipeline.train_stage(
        interrupted_config,
        stage="base",
        requested_device="cpu",
        resume=periodic_path,
    )
    assert resumed["status"] == "complete"
    continuous_config = configured(tmp_path / "continuous")
    continuous = pipeline.train_stage(
        continuous_config, stage="base", requested_device="cpu"
    )
    assert continuous["status"] == "complete"
    resumed_checkpoint = torch.load(
        tmp_path / "interrupted" / "best_dev.pt", map_location="cpu", weights_only=True
    )
    continuous_checkpoint = torch.load(
        tmp_path / "continuous" / "best_dev.pt", map_location="cpu", weights_only=True
    )
    assert resumed_checkpoint["global_step"] == continuous_checkpoint["global_step"] == 2
    for key, expected in continuous_checkpoint["student_state_dict"].items():
        torch.testing.assert_close(
            resumed_checkpoint["student_state_dict"][key], expected, atol=0, rtol=0
        )
    resumed_step = json.loads(
        (tmp_path / "interrupted" / "latest_train_step.json").read_text(encoding="utf-8")
    )
    continuous_step = json.loads(
        (tmp_path / "continuous" / "latest_train_step.json").read_text(encoding="utf-8")
    )
    assert resumed_step["batch_identity_sha256"] == continuous_step[
        "batch_identity_sha256"
    ]
    assert resumed_step["loss"] == continuous_step["loss"]
