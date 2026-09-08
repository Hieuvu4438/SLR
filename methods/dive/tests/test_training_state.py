from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import torch
from torch import nn

from dive.training.state import (
    CheckpointError,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _components(initial_state=None):
    model = nn.Linear(3, 2)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    return model, optimizer, scheduler


def _random_step(model, optimizer, scheduler):
    python_value = random.random()
    numpy_value = float(np.random.random())
    inputs = torch.randn(4, 3)
    target = torch.randn(4, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach()), python_value, numpy_value


def _assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    elif isinstance(first, Mapping):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(53)
    initial = nn.Linear(3, 2).state_dict()
    uninterrupted, optimizer_a, scheduler_a = _components(initial)
    random.seed(59)
    np.random.seed(59)
    torch.manual_seed(59)
    _random_step(uninterrupted, optimizer_a, scheduler_a)
    checkpoint = tmp_path / "student.pt"
    save_training_checkpoint(
        checkpoint,
        model=uninterrupted,
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        scaler=None,
        epoch=0,
        global_step=1,
        best_dev={"endpoint": 0.5, "step": 0},
        sampler_state={"plan_hash": "plan", "cursor": 1, "quota": {"s0": 1}},
        fingerprints={"bank": "bank", "reference": "reference", "data": "data"},
        config_hash="config",
        git_revision="fixture_revision",
        optimizer_manifest={"groups": ["projector", "pose"]},
    )
    continuation_a = _random_step(uninterrupted, optimizer_a, scheduler_a)

    resumed, optimizer_b, scheduler_b = _components(initial)
    state = load_training_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        scaler=None,
        expected_config_hash="config",
        expected_fingerprints={"bank": "bank", "reference": "reference", "data": "data"},
    )
    continuation_b = _random_step(resumed, optimizer_b, scheduler_b)
    assert state.global_step == 1 and state.sampler_state["cursor"] == 1
    assert continuation_b == pytest.approx(continuation_a)
    for left, right in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    _assert_nested_equal(optimizer_a.state_dict(), optimizer_b.state_dict())
    _assert_nested_equal(scheduler_a.state_dict(), scheduler_b.state_dict())


def test_resume_rejects_fingerprint_and_checksum_mismatch(tmp_path):
    model, optimizer, scheduler = _components()
    checkpoint = tmp_path / "student.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=0,
        global_step=0,
        best_dev={},
        sampler_state={},
        fingerprints={"bank": "one"},
        config_hash="config",
        git_revision="fixture_revision",
        optimizer_manifest={},
    )
    with pytest.raises(CheckpointError, match="fingerprint mismatch"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_config_hash="config",
            expected_fingerprints={"bank": "two"},
        )
    with checkpoint.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(CheckpointError, match="checksum mismatch"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_config_hash="config",
            expected_fingerprints={"bank": "one"},
        )
