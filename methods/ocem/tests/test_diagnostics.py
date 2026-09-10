from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from ocem.diagnostics.cico_stage_a import (
    StageADiagnosticError,
    _selected_inputs,
    _verify_text_records,
)
from ocem.diagnostics.concentration import (
    adjusted_paired_effect,
    bootstrap_mean_ci,
    independent_concentration,
    weighted_quantile,
)
from ocem.diagnostics.errors import equivalence_recall, select_r1_errors
from ocem.diagnostics.shortcuts import (
    assign_quantile_bin,
    deterministic_negative_overlaps,
    lexical_jaccard,
    quantile_bin_edges,
)
from ocem.scoring.geometry import build_support_geometry


def test_independent_concentration_uses_weighted_atom_density() -> None:
    geometry = build_support_geometry([[0, 2], [1, 3], [8, 10]])
    concentrated = np.array([[0.9, 0.85, -0.2], [0.88, 0.9, -0.2]])
    distributed = np.array([[0.9, -0.2, 0.8], [-0.2, 0.9, 0.8]])
    first = independent_concentration(
        concentrated, geometry.A, geometry.w, geometry.q
    )
    second = independent_concentration(distributed, geometry.A, geometry.w, geometry.q)
    assert first.concentration_p95 > second.concentration_p95
    assert first.excess_kappa > second.excess_kappa
    assert first.real_mass + first.null_mass == pytest.approx(1.0)
    assert weighted_quantile(np.array([1.0, 9.0]), np.array([0.96, 0.04]), 0.95) == 1.0


def test_bootstrap_and_adjustment_are_deterministic() -> None:
    values = np.arange(1.0, 9.0)
    first = bootstrap_mean_ci(values, replicates=1000, seed=7)
    second = bootstrap_mean_ci(values, replicates=1000, seed=7)
    assert first == second
    adjusted = adjusted_paired_effect(
        values,
        np.column_stack((np.arange(8), np.ones(8))),
        ["varying", "constant"],
        replicates=1000,
        seed=9,
    )
    assert adjusted["dropped_constant_controls"] == ["constant"]
    assert adjusted["design_rank"] == adjusted["design_columns"]


def test_error_selection_uses_canonical_ties_and_keeps_duplicate_queries() -> None:
    ids = ["b", "a", "c"]
    scores = np.array(
        [
            [0.4, 0.9, 0.1],
            [0.4, 0.8, 0.2],
            [0.1, 0.1, 0.7],
        ]
    )
    errors = select_r1_errors(scores, video_ids=ids, text_ids=ids, direction="T2V")
    assert errors[0]["query_id"] == "b"
    assert errors[0]["false_video_id"] == "a"
    hashes = {"a": "same", "b": "same", "c": "other"}
    result = equivalence_recall(
        scores,
        video_ids=ids,
        text_ids=ids,
        caption_hash_by_id=hashes,
        direction="T2V",
    )
    assert result["queries"] == 3
    assert result["R1"] == 100.0


def test_shortcut_controls_are_train_determined() -> None:
    assert lexical_jaccard("A b", "b c") == pytest.approx(1 / 3)
    edges = quantile_bin_edges([0, 1, 2, 3, 4, 5, 6, 7])
    assert assign_quantile_bin(0.0, edges) == 0
    ids = ["a", "b", "c"]
    captions = {"a": "one two", "b": "two three", "c": "four"}
    assert deterministic_negative_overlaps(ids, captions, seed=3) == (
        deterministic_negative_overlaps(ids, captions, seed=3)
    )


def test_stage_a_selection_keeps_feature_and_support_indices_aligned(tmp_path) -> None:
    agnostic, adapted, temporal = (
        tmp_path / "agnostic",
        tmp_path / "adapted",
        tmp_path / "temporal",
    )
    for path in (agnostic, adapted, temporal):
        path.mkdir()
    values = np.arange(5 * 1024, dtype=np.float32).reshape(5, 1024)
    for root, offset in ((agnostic, 0.0), (adapted, 10.0)):
        with (root / "sample.pkl").open("wb") as handle:
            pickle.dump({"feature": values + offset}, handle)
    (temporal / "sample.json").write_text(
        json.dumps({"rf_start": [0, 1, 2, 3, 4], "rf_end": [16, 17, 18, 19, 20]})
    )
    selected, valid, intervals = _selected_inputs(
        "sample",
        agnostic_dir=agnostic,
        adapted_dir=adapted,
        temporal_dir=temporal,
        feature_len=3,
        alpha=0.75,
    )
    assert valid.tolist() == [True, True, True]
    assert np.array_equal(intervals, [[0, 16], [2, 18], [4, 20]])
    assert np.array_equal(selected, values[[0, 2, 4]] + 2.5)


def test_stage_a_text_resource_requires_exact_manifest_order_and_original_text() -> None:
    records = [
        {"sample_id": "a", "caption_raw": "first"},
        {"sample_id": "b", "caption_raw": "second"},
    ]
    mapping = {
        "a": {"video_name": "a", "ori_text": "first", "text": "one"},
        "b": {"video_name": "b", "ori_text": "second", "text": "two"},
    }
    _verify_text_records(records, mapping)
    with pytest.raises(StageADiagnosticError, match="order"):
        _verify_text_records(records, {"b": mapping["b"], "a": mapping["a"]})
