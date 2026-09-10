from __future__ import annotations

import torch

from ocem.evaluation.scoring import (
    build_text_index,
    build_video_index,
    score_directional_full_gallery,
)


def _indexes():
    videos = build_video_index(
        ["v0", "v1", "v2"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        torch.ones(3, 1, dtype=torch.bool),
        torch.tensor([[[0.0, 1.0]], [[0.0, 2.0]], [[1.0, 3.0]]]),
    )
    texts = build_text_index(
        ["t2", "t0", "t1", "t3"],
        torch.tensor([[0.2, 0.8], [1.0, 0.0], [0.0, 1.0], [0.3, 0.7]]),
        torch.ones(4, 1, dtype=torch.bool),
    )
    return videos, texts


def _scorer(video, video_valid, intervals, text, text_valid):
    del video_valid, intervals, text_valid
    base = video @ text.T
    return base, base + video[:, :1] - text[:, :1].T


def test_block_scorer_matches_dense_and_covers_every_pair() -> None:
    videos, texts = _indexes()
    result = score_directional_full_gallery(
        videos, texts, _scorer, video_block_size=2, text_block_size=3
    )
    expected_t2v, expected_v2t = _scorer(
        videos.representations,
        videos.valid,
        videos.intervals,
        texts.representations,
        texts.valid,
    )
    assert torch.equal(result.scores_t2v, expected_t2v)
    assert torch.equal(result.scores_v2t, expected_v2t)
    assert result.scored_pairs == result.expected_pairs == 12


def test_t2v_scores_cannot_depend_on_candidate_video_captions() -> None:
    annotations = {
        "v0": "first candidate caption",
        "v1": "second candidate caption",
        "v2": "third candidate caption",
    }
    videos, texts = _indexes()
    before = score_directional_full_gallery(
        videos, texts, _scorer, video_block_size=1, text_block_size=2
    ).scores_t2v
    annotations.clear()
    annotations.update({"v0": "changed", "v1": "", "v2": "deleted meaning"})
    after = score_directional_full_gallery(
        videos, texts, _scorer, video_block_size=3, text_block_size=4
    ).scores_t2v
    assert torch.equal(before, after)
    assert not hasattr(videos, "captions")
