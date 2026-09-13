from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import method1.trainer as trainer_module
from method1.checkpoints import (
    CheckpointError,
    DevSelection,
    atomic_torch_save,
    make_training_checkpoint,
    validate_resume_identity,
)
from method1.config import load_config
from method1.distributed import DistributedRuntime
from method1.optimizer import build_upret_optimizer
from method1.schemas import LocalEncoding
from method1.trainer import Method1TrainModel, complete_optimizer_step


CONFIG = Path("methods/sssc/configs/method1/ph_local.yaml")


class TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip = nn.Module()
        self.clip.projection = nn.Linear(2, 2)
        self.clip.LayerNorm = nn.LayerNorm(2)
        self.clip.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.head = nn.Linear(2, 1)


def test_optimizer_matches_upstream_group_and_hyperparameter_contract() -> None:
    config = load_config(CONFIG)
    result = build_upret_optimizer(TinyStudent(), config, total_steps=17)
    assert result.optimizer.__class__.__name__ == "BertAdam"
    assert [group["name"] for group in result.parameter_groups] == [
        "decay_clip",
        "decay_nonclip",
        "no_decay_clip",
        "no_decay_nonclip",
    ]
    assert {name for group in result.parameter_groups for name in group["parameter_names"]} == {
        name for name, _ in TinyStudent().named_parameters()
    }
    for group in result.optimizer.param_groups:
        assert group["schedule"] == "warmup_cosine"
        assert group["warmup"] == pytest.approx(0.1)
        assert group["t_total"] == 17
        assert group["b1"] == pytest.approx(0.9)
        assert group["b2"] == pytest.approx(0.98)


def test_dev_selector_uses_r1_then_r5_then_earliest_step() -> None:
    earlier = DevSelection.from_metrics(
        {"T2V": {"R1": 10, "R5": 20}, "V2T": {"R1": 30, "R5": 40}}, 5
    )
    later_tie = DevSelection.from_metrics(
        {"T2V": {"R1": 20, "R5": 30}, "V2T": {"R1": 20, "R5": 30}}, 9
    )
    better_r5 = DevSelection.from_metrics(
        {"T2V": {"R1": 20, "R5": 35}, "V2T": {"R1": 20, "R5": 35}}, 10
    )
    assert not later_tie.beats(earlier)
    assert better_r5.beats(earlier)


def test_checkpoint_is_atomic_and_resume_identity_is_strict(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    student = TinyStudent()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    metrics = {
        "T2V": {"R1": 10.0, "R5": 20.0},
        "V2T": {"R1": 30.0, "R5": 40.0},
    }
    hashes = {"dataset": "abc", "teacher": None, "cache": None}
    checkpoint = make_training_checkpoint(
        config=config,
        student=student,
        optimizer=optimizer,
        epoch=2,
        next_batch_index=3,
        global_step=11,
        sampler_state={"epoch": 2, "seed": 42},
        artifact_hashes=hashes,
        implementation_revision="deadbeef",
        dev_metrics=metrics,
    )
    destination = tmp_path / "last.pt"
    atomic_torch_save(checkpoint, destination)
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    validate_resume_identity(loaded, config=config, artifact_hashes=hashes)
    with pytest.raises(CheckpointError, match="artifact_hashes"):
        validate_resume_identity(loaded, config=config, artifact_hashes={"dataset": "changed"})


def test_train_wrapper_combines_base_and_shared_auxiliary(monkeypatch) -> None:
    config = load_config(CONFIG)
    student = TinyStudent()
    auxiliary = replace(config.auxiliary, arm="span_shared", support_mode="shared")
    video = student.head.weight.reshape(1, 1, 2).expand(1, 3, 2)
    encoding = LocalEncoding(
        video_raw=video,
        video_ignore_raw=torch.tensor([[True, False, False]]),
        text_raw=torch.ones(1, 2, 2),
        text_valid=torch.ones(1, 2, dtype=torch.bool),
        text_aug_raw=torch.ones(1, 2, 2),
        text_aug_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    monkeypatch.setattr(trainer_module, "encode_local", lambda _student, _batch: encoding)
    monkeypatch.setattr(
        trainer_module,
        "baseline_loss_from_local",
        lambda core, enc, **kwargs: core.head.weight.square().sum(),
    )
    wrapper = Method1TrainModel(
        student, auxiliary, DistributedRuntime(), baseline_seed=config.seed
    )
    batch = {
        "x_ref": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "q_pos": torch.tensor([[[[1.0, 0.0]]]]),
        "q_neg": torch.tensor([[[[0.0, 1.0]]]]),
        "edit_valid": torch.tensor([[[True]]]),
        "confidence": torch.tensor([[[0.25]]]),
    }
    result = wrapper(batch, optimizer_step=4, epoch=2)
    assert set(result) == {
        "loss",
        "base",
        "auxiliary",
        "auxiliary_numerator",
        "auxiliary_weight_count",
    }
    assert result["auxiliary_weight_count"].item() == pytest.approx(1.0)
    result["loss"].backward()
    assert student.head.weight.grad is not None


def test_optimizer_step_clips_and_clamps_logit_scale() -> None:
    student = TinyStudent()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    student.head.weight.square().sum().backward()
    norm = complete_optimizer_step(student, optimizer, max_grad_norm=1.0)
    assert torch.isfinite(norm)
    assert student.clip.logit_scale.item() == pytest.approx(float(torch.log(torch.tensor(100.0))))
