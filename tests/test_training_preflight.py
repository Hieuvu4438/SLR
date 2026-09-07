from __future__ import annotations

import pytest
import torch
from torch import nn

from elsc.train import _keep_objective
from elsc.training_preflight import _stress_batch_indices


def test_preflight_stress_batch_prioritizes_evidence_without_duplicates():
    pair_ids = ["a", "b", "c", "d", "e"]
    indexes = _stress_batch_indices(pair_ids, {"c", "e"}, 4)
    assert indexes == [2, 4, 0, 1]
    assert len(indexes) == len(set(indexes))


def test_preflight_baseline_batch_retains_manifest_order():
    assert _stress_batch_indices(["a", "b", "c"], set(), 2) == [0, 1]


def test_preflight_rejects_batch_larger_than_manifest():
    with pytest.raises(ValueError, match="manifest is smaller"):
        _stress_batch_indices(["a"], {"a"}, 2)


class _KeepBridge:
    @staticmethod
    def score(video, text, *, objective):
        assert objective
        scores = video @ text.T
        return scores, scores

    @staticmethod
    def mixed_score(i2t, t2i, dual_mix):
        return dual_mix * i2t + (1.0 - dual_mix) * t2i


class _KeepModel(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bridge = _KeepBridge()

    def encode_video(self, h, valid):
        del valid
        return self.scale * h.mean(dim=1), h

    def encode_text(self, ids, segments, mask):
        del segments, mask
        return self.scale * ids.float()


def test_keep_objective_detaches_teacher_and_updates_student():
    student = _KeepModel(0.8)
    teacher = _KeepModel(1.2)
    h = torch.tensor([[[1.0, 0.0]], [[0.5, 1.0]]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    ids = torch.tensor([[1, 0], [0, 1]])
    segments = torch.zeros_like(ids)
    mask = torch.ones_like(ids)
    video, _ = student.encode_video(h, valid)
    text = student.encode_text(ids, segments, mask)

    loss = _keep_objective(
        student,
        teacher,
        h,
        valid,
        (ids, segments, mask),
        video,
        text,
        dual_mix=0.5,
        temperature=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert student.scale.grad is not None
    assert torch.isfinite(student.scale.grad)
    assert teacher.scale.grad is None
