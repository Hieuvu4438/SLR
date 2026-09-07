from __future__ import annotations

import pytest

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
