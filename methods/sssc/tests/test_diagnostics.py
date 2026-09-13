from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import torch

from method1.diagnostics import (
    _corrupt_and_reencode,
    _distribution,
    _feature_removal_positions,
    _support_metrics,
    loss_gradient_cosine,
)
from method1.reference import freeze_reference
from method1.utils import atomic_jsonl_dump


def test_support_diagnostic_matches_shared_independent_fixture() -> None:
    result = _support_metrics(
        np.eye(2, dtype=np.float32),
        np.array([True, True]),
        [10, 20],
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        tau=0.07,
        margin=0.05,
    )
    assert result["shared_margin"] == pytest.approx(1.0, abs=2e-6)
    assert result["independent_margin"] == pytest.approx(0.0, abs=2e-6)
    assert result["peak_displacement_feature_rows"] == 10
    assert result["support_overlap"] < 1e-5
    assert result["support_jsd"] == pytest.approx(np.log(2.0), abs=2e-5)


def test_diagnostic_distribution_and_atomic_records(tmp_path: Path) -> None:
    assert _distribution([])["mean"] is None
    summary = _distribution([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    path = tmp_path / "records.jsonl"
    atomic_jsonl_dump([{"id": 1}, {"id": 2}], path)
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == [1, 2]


class ContextualReference(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))

    def get_video_feat(self, video, mask, **kwargs):
        local = video[:, :2, :, 0].transpose(1, 2)
        context = local + local.mean(dim=1, keepdim=True)
        cls = torch.zeros(video.shape[0], 1, 2, device=video.device)
        return mask, torch.cat((cls, context), dim=1), cls[:, 0]


def test_selected_and_random_feature_removal_reencode_corrupted_inputs() -> None:
    reference = freeze_reference(ContextualReference())
    features = torch.zeros(1, 2, 64, 1)
    features[0, :, :4, 0] = torch.tensor(
        [[4.0, 1.0, 0.0, 1.0], [0.0, 1.0, 4.0, 2.0]]
    )
    mask = torch.ones(1, 65, dtype=torch.bool)
    mask[:, 1:5] = False
    encoded, valid = _corrupt_and_reencode(reference, features, mask, ((),))
    removal = _feature_removal_positions(
        encoded[0].numpy(),
        valid[0].numpy(),
        [np.array([1.0, 0.0], dtype=np.float32)],
        [10, 20, 30, 40, *([-1] * 60)],
        tau=0.07,
        seed_parts=(42, "dev", "video"),
    )
    assert removal["informative"] is True
    assert removal["removal_count"] == 1
    assert removal["selected_positions"] != removal["random_positions"]
    assert removal["selected_feature_rows"] == (10,)
    corrupted, corrupted_valid = _corrupt_and_reencode(
        reference,
        features,
        mask,
        (removal["selected_positions"], removal["random_positions"]),
    )
    assert corrupted_valid.sum(dim=1).tolist() == [3, 3]
    assert not torch.equal(corrupted[0], encoded[0])
    assert not torch.equal(corrupted[0], corrupted[1])


def test_loss_gradient_cosine_uses_separate_autograd_grad_calls() -> None:
    left = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    right = torch.nn.Parameter(torch.tensor([3.0]))
    base = left.square().sum() + right.square().sum()
    auxiliary = -left.square().sum() + 2.0 * right.square().sum()
    result = loss_gradient_cosine([("left", left), ("right", right)], base, auxiliary)
    expected_base = torch.tensor([2.0, 4.0, 6.0])
    expected_auxiliary = torch.tensor([-2.0, -4.0, 12.0])
    expected = torch.nn.functional.cosine_similarity(
        expected_base, expected_auxiliary, dim=0
    )
    assert result["base_auxiliary_gradient_cosine"] == pytest.approx(float(expected))
    assert result["parameter_tensor_count"] == 2
    assert result["jointly_active_parameter_tensor_count"] == 2
