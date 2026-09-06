from __future__ import annotations

import numpy as np
import pytest
import torch

from elsc.evaluation.cico_eval import (
    EvaluationContractError,
    evaluate_score_matrix,
    singleton_positive_maps,
)
from elsc.evaluation.runtime import GalleryEncodings, encode_gallery, score_gallery_blockwise
from elsc.upstream.cico_bridge import TextEncoding, VideoEncoding


def test_asymmetric_orientation_fixture():
    scores = np.array([[9, 8, 0], [10, 7, 0], [0, 0, 9]], dtype=np.float32)
    videos = ["v0", "v1", "v2"]
    texts = ["t0", "t1", "t2"]
    v2t, t2v = singleton_positive_maps(videos, texts)
    result = evaluate_score_matrix(
        scores, video_ids=videos, text_ids=texts, video_to_text=v2t, text_to_video=t2v
    )
    assert result["V2T"]["R1"] == pytest.approx(200 / 3)
    assert result["T2V"]["R1"] == pytest.approx(100 / 3)
    assert result["per_query"]["V2T"][1]["ranked_candidate_ids"][0] == "t0"
    assert result["per_query"]["T2V"][0]["ranked_candidate_ids"][0] == "v1"


def test_gallery_shape_and_positive_mapping_are_enforced():
    with pytest.raises(EvaluationContractError):
        evaluate_score_matrix(
            np.ones((2, 2)),
            video_ids=["v"],
            text_ids=["a", "b"],
            video_to_text={"v": ["a"]},
            text_to_video={"a": ["v"], "b": ["v"]},
        )


def test_singleton_ties_match_cico_duplicate_rank_behavior():
    scores = np.ones((2, 2), dtype=np.float32)
    videos = ["v0", "v1"]
    texts = ["t0", "t1"]
    v2t, t2v = singleton_positive_maps(videos, texts)
    result = evaluate_score_matrix(
        scores, video_ids=videos, text_ids=texts, video_to_text=v2t, text_to_video=t2v
    )
    # CiCo emits both rank positions for each tied positive: [0,1,0,1].
    assert result["V2T"]["R1"] == 50.0
    assert result["V2T"]["cols"] == [0, 1, 0, 1]
    assert result["diagnostic_best_positive"]["V2T"]["R1"] == 100.0
    assert result["tie_stats"]["V2T_queries_with_positive_tie"] == 2


class _BlockBridge:
    @staticmethod
    def score(video, text, objective=True):
        score = video.tokens[:, 0] @ text.tokens[:, 0].T
        return score, score + 0.25

    @staticmethod
    def mixed_score(i2t, t2i, dual_mix):
        return dual_mix * i2t + (1 - dual_mix) * t2i


def test_block_scoring_matches_unblocked_scoring():
    video_tokens = torch.randn(5, 1, 4)
    text_tokens = torch.randn(7, 1, 4)
    gallery = GalleryEncodings(
        VideoEncoding(torch.ones(5, 1), video_tokens, torch.zeros(5, 4)),
        TextEncoding(torch.ones(7, 1), text_tokens, torch.zeros(7, 4)),
        [f"v{i}" for i in range(5)],
        [f"t{i}" for i in range(7)],
    )
    blocked = score_gallery_blockwise(
        _BlockBridge(),
        gallery,
        device=torch.device("cpu"),
        dual_mix=0.3,
        video_block=2,
        text_block=3,
    )
    score = video_tokens[:, 0] @ text_tokens[:, 0].T
    expected = 0.3 * score + 0.7 * (score + 0.25)
    assert np.allclose(blocked, expected.numpy())


class _GalleryModel:
    @staticmethod
    def eval():
        return None

    @staticmethod
    def encode_video(h, valid):
        return VideoEncoding(valid, h, h[:, 0]), h

    @staticmethod
    def encode_text(ids, segments, mask):
        return TextEncoding(mask, ids.float().unsqueeze(-1), ids[:, 0].float().unsqueeze(-1))


def test_gallery_deduplicates_video_and_preserves_multi_positive_mapping():
    batch = {
        "h": torch.tensor([[[1.0]], [[1.0]], [[2.0]]]),
        "valid": torch.ones(3, 1, dtype=torch.bool),
        "clean_text": (
            torch.tensor([[1], [2], [3]]),
            torch.zeros(3, 1, dtype=torch.long),
            torch.ones(3, 1, dtype=torch.long),
        ),
        "video_id": ["v0", "v0", "v1"],
        "caption_id": ["t0", "t1", "t2"],
    }
    gallery = encode_gallery(_GalleryModel(), [batch], torch.device("cpu"))
    assert gallery.video_ids == ["v0", "v1"]
    assert gallery.text_ids == ["t0", "t1", "t2"]
    assert gallery.videos.tokens.shape[0] == 2
    assert gallery.video_to_text == {"v0": ["t0", "t1"], "v1": ["t2"]}
    assert gallery.text_to_video == {"t0": ["v0"], "t1": ["v0"], "t2": ["v1"]}
