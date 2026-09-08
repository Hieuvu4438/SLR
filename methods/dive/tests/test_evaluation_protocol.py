from __future__ import annotations

import json

import numpy as np
import pytest

from dive.eval.protocol import (
    EvaluationProtocolError,
    calibrate_locked_checkpoint,
    duplicate_query_ceiling,
    opportunity_bound,
    write_calibration_selection,
)


VIDEO_IDS = ("v0", "v1")
TEXT_IDS = ("t0", "t1")
RELEVANCE = {"v0": ("t0",), "v1": ("t1",)}


def _scores():
    baseline = np.asarray([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64)
    correction = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    return baseline, correction


def test_gamma_calibration_uses_locked_checkpoint_and_smallest_gamma_tie(tmp_path):
    baseline, correction = _scores()
    selection = calibrate_locked_checkpoint(
        baseline,
        correction,
        VIDEO_IDS,
        TEXT_IDS,
        RELEVANCE,
        gamma_grid=(0.2, 0.0, 0.1, 0.05),
        topk=(1,),
    )
    assert selection.selected_gamma == 0.1
    assert selection.no_added_value_on_dev is False
    assert [item.gamma for item in selection.candidates] == [0.0, 0.05, 0.1, 0.2]
    assert selection.candidates[0].metrics.v2t.recall[1] == 0.0
    path = write_calibration_selection(
        tmp_path / "selection.json",
        selection,
        locked_checkpoint_path="winner.pt",
        locked_checkpoint_sha256="a" * 64,
        locked_epoch=3,
        locked_optimizer_step=30,
        checkpoint_selection_gamma=0.1,
        config_hash="config_sha",
        video_ids=VIDEO_IDS,
        text_ids=TEXT_IDS,
        video_to_text_positives=RELEVANCE,
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["selection_order"] == (
        "checkpoint_at_fixed_gamma_then_gamma_for_locked_checkpoint"
    )
    assert artifact["checkpoint"]["epoch"] == 3
    assert artifact["calibration"]["selected_gamma"] == 0.1
    assert len(artifact["dev_ids_sha256"]) == 64


def test_gamma_zero_is_preserved_and_test_split_calibration_is_rejected():
    baseline = np.eye(2, dtype=np.float64)
    correction = np.asarray([[0.8, -0.4], [-0.2, 0.7]])
    selection = calibrate_locked_checkpoint(
        baseline, correction, VIDEO_IDS, TEXT_IDS, RELEVANCE, gamma_grid=(0.0, 0.2)
    )
    assert selection.selected_gamma == 0.0
    assert selection.no_added_value_on_dev is True
    with pytest.raises(EvaluationProtocolError, match="dev-only"):
        calibrate_locked_checkpoint(
            baseline,
            correction,
            VIDEO_IDS,
            TEXT_IDS,
            RELEVANCE,
            split="test",
        )


def test_opportunity_bound_excludes_gaps_above_two_gamma_in_both_directions():
    baseline, _ = _scores()
    report = opportunity_bound(
        baseline,
        VIDEO_IDS,
        TEXT_IDS,
        RELEVANCE,
        gamma_grid=(0.0, 0.04, 0.05),
    )
    for direction in (report.v2t, report.t2v):
        assert direction.baseline_errors == 2
        assert direction.repairable_error_count[0.0] == 0
        assert direction.repairable_error_count[0.04] == 0
        assert direction.repairable_error_count[0.05] == 2
        assert direction.upper_bound_absolute_r1_gain[0.05] == 1.0


def test_opportunity_uses_best_annotated_positive_not_diagonal():
    scores = np.asarray([[0.9, 0.8, 0.1], [0.5, 0.4, 0.3]])
    relevance = {"v0": ("t1", "t2"), "v1": ("t0",)}
    report = opportunity_bound(
        scores,
        ("v0", "v1"),
        ("t0", "t1", "t2"),
        relevance,
        gamma_grid=(0.0, 0.05),
    )
    assert report.v2t.error_gaps["v0"] == pytest.approx(0.1)
    assert report.v2t.repairable_error_count[0.05] == 1


def test_duplicate_group_ceiling_keeps_queries_and_uses_annotation_sets():
    ceiling = duplicate_query_ceiling(
        {"t0": "same text", "t1": " same   text ", "t2": "different"},
        {"t0": ("v0",), "t1": ("v1",), "t2": ("v2",)},
    )
    assert ceiling.num_queries == 3
    assert ceiling.maximum_top1_hits == 2
    assert ceiling.r1_ceiling == pytest.approx(2 / 3)
    assert ceiling.groups[0].query_ids == ("t0", "t1")
    assert ceiling.groups[0].maximum_top1_hits == 1
