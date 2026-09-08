from __future__ import annotations

import torch

from dive.data import build_synthetic_fixture
from dive.eval import evaluate_retrieval
from dive.models import evidence_score_block


def test_fixture_contains_all_required_smoke_cases():
    fixture = build_synthetic_fixture()
    assert fixture.video_mask.sum(1).unique().numel() > 1
    assert fixture.text_mask.sum(1).unique().numel() > 1
    assert {item.support_status for item in fixture.contrasts} == {"accepted", "tiny_text_distance"}
    assert fixture.excluded_negatives
    (video_a, text_a), (video_b, text_b) = fixture.tie_pair
    assert video_a == video_b
    row = fixture.video_ids.index(video_a)
    assert fixture.baseline_scores[row, fixture.text_ids.index(text_a)] == fixture.baseline_scores[
        row, fixture.text_ids.index(text_b)
    ]


def test_fixture_has_planted_retrieval_signal_without_claiming_a_benchmark():
    fixture = build_synthetic_fixture()
    score, valid = evidence_score_block(
        fixture.video_evidence,
        fixture.text_units,
        fixture.video_mask,
        fixture.text_mask,
    )
    assert valid.all()
    metrics = evaluate_retrieval(score.detach().numpy(), fixture.video_ids, fixture.text_ids, fixture.relevance)
    assert metrics.v2t.recall[1] == 1.0
    assert metrics.t2v.recall[1] == 1.0
    assert torch.isfinite(score).all()
