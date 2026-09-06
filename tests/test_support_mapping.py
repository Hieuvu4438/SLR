from __future__ import annotations

import numpy as np
import torch

from elsc.data.views import canonical_view, jittered_view, materialize_view
from elsc.mining.teacher_align import interval_iou, reliability, select_support


def test_temporal_views_are_deterministic_and_use_dense_ids():
    a = canonical_view(100, 8)
    b1, independent1 = jittered_view(100, 8, global_seed=42, video_id="v1")
    b2, independent2 = jittered_view(100, 8, global_seed=42, video_id="v1")
    assert independent1 and independent2
    assert np.array_equal(b1.indices, b2.indices)
    assert len(np.unique(b1.indices)) == 8
    assert np.all(np.diff(b1.indices) > 0)
    h, valid, dense = materialize_view(torch.randn(100, 4), a, 8)
    assert h.shape == (8, 4) and valid.all()
    assert dense.tolist() == a.indices.tolist()


def test_short_video_cannot_fake_independent_view():
    view, independent = jittered_view(3, 8, global_seed=1, video_id="short")
    assert not independent
    assert view.indices.tolist() == [0, 1, 2]


def test_support_selector_and_reliability():
    p = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    q = torch.tensor([[0.05, 0.45], [0.05, 0.45], [0.45, 0.05], [0.45, 0.05]])
    dense = torch.tensor([10, 20, 30, 40])
    support = select_support(
        p,
        q,
        dense,
        1,
        mass_min=0.8,
        min_support_tokens=2,
        max_duration_fraction=0.5,
        confidence_min=0.8,
    )
    assert support is not None
    assert support.dense_indices == (10, 20)
    assert support.interval == (10, 21)
    iou, rho = reliability(support, support, occurrence_count=5)
    assert iou == 1.0 and rho >= 0.89
    assert interval_iou((0, 4), (2, 6)) == 1 / 3
