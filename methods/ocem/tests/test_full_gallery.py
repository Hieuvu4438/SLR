from __future__ import annotations

import numpy as np
import pytest

from ocem.evaluation.full_gallery import EvaluationError, evaluate_full_gallery


VIDEO_IDS = ["v2", "v0", "v1", "v3"]
TEXT_IDS = ["t1", "t3", "t0", "t2"]
TEXT_TO_VIDEO = {"t0": "v0", "t1": "v1", "t2": "v2", "t3": "v3"}
VIDEO_TO_TEXT = {"v0": "t0", "v1": "t1", "v2": "t2", "v3": "t3"}


def test_asymmetric_permuted_full_gallery_uses_correct_axes_and_ids() -> None:
    # Columns are text queries for T2V; rows are video queries for V2T.
    t2v = np.array(
        [
            [0.1, 0.2, 0.3, 0.9],  # v2 is the target for t2 (last column)
            [0.2, 0.1, 0.9, 0.2],  # v0 is the target for t0
            [0.9, 0.2, 0.1, 0.3],  # v1 is the target for t1
            [0.3, 0.9, 0.2, 0.1],  # v3 is the target for t3
        ],
        dtype=np.float32,
    )
    v2t = np.array(
        [
            [0.1, 0.2, 0.3, 0.9],  # v2 -> t2
            [0.2, 0.1, 0.9, 0.3],  # v0 -> t0
            [0.9, 0.2, 0.1, 0.3],  # v1 -> t1
            [0.2, 0.9, 0.3, 0.1],  # v3 -> t3
        ],
        dtype=np.float32,
    )
    report = evaluate_full_gallery(
        t2v,
        v2t,
        video_ids=VIDEO_IDS,
        text_ids=TEXT_IDS,
        text_to_video=TEXT_TO_VIDEO,
        video_to_text=VIDEO_TO_TEXT,
    )
    assert report["T2V"]["R1"] == 100.0
    assert report["V2T"]["R1"] == 100.0
    assert report["coverage"]["t2v_scores"] == 16
    assert report["coverage"]["fraction"] == 1.0


def test_exact_tie_uses_canonical_id_and_reports_unweighted_sensitivity() -> None:
    scores = np.eye(4, dtype=np.float32)
    # Query t0 targets v0, but v0 and lexicographically smaller v1 tie.
    t0_column = TEXT_IDS.index("t0")
    scores[:, t0_column] = 0.0
    scores[VIDEO_IDS.index("v0"), t0_column] = 1.0
    scores[VIDEO_IDS.index("v1"), t0_column] = 1.0
    report = evaluate_full_gallery(
        scores,
        scores.copy(),
        video_ids=VIDEO_IDS,
        text_ids=TEXT_IDS,
        text_to_video=TEXT_TO_VIDEO,
        video_to_text=VIDEO_TO_TEXT,
    )
    assert report["T2V"]["primary_ranks"][t0_column] == 1
    assert report["T2V"]["tie_queries"] >= 1
    sensitivity = report["T2V"]["tie_sensitivity"]["R1"]
    assert sensitivity["expected"] == pytest.approx(
        (sum(rank == 1 for rank in report["T2V"]["primary_ranks"][:-2]) + 0.5) * 25
    )
    assert sensitivity["optimistic"] >= sensitivity["expected"]
    assert sensitivity["expected"] >= sensitivity["pessimistic"]


def test_duplicate_caption_queries_remain_separate() -> None:
    scores = np.eye(3, dtype=np.float32)
    report = evaluate_full_gallery(
        scores,
        scores,
        video_ids=["v0", "v1", "v2"],
        text_ids=["duplicate-a", "duplicate-b", "other"],
        text_to_video={"duplicate-a": "v0", "duplicate-b": "v1", "other": "v2"},
        video_to_text={"v0": "duplicate-a", "v1": "duplicate-b", "v2": "other"},
    )
    assert report["T2V"]["queries"] == 3
    assert report["V2T"]["queries"] == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_target", "absent"),
        ("missing_query", "every text query"),
        ("nonfinite", "non-finite"),
        ("wrong_shape", "shape"),
    ],
)
def test_invalid_protocol_or_incomplete_gallery_fails_closed(change: str, message: str) -> None:
    scores = np.eye(4, dtype=np.float32)
    text_to_video = dict(TEXT_TO_VIDEO)
    if change == "missing_target":
        text_to_video["t0"] = "missing"
    elif change == "missing_query":
        text_to_video.pop("t0")
    elif change == "nonfinite":
        scores[0, 0] = np.nan
    elif change == "wrong_shape":
        scores = scores[:3]
    with pytest.raises(EvaluationError, match=message):
        evaluate_full_gallery(
            scores,
            scores,
            video_ids=VIDEO_IDS,
            text_ids=TEXT_IDS,
            text_to_video=text_to_video,
            video_to_text=VIDEO_TO_TEXT,
        )
