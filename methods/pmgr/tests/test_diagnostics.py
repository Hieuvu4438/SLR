from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pmgr.diagnostics import _gradient_attribution, weak_performance_diagnostic
from pmgr.metrics import evaluate_protocol


def test_gradient_attribution_names_largest_performance_and_group():
    reference = torch.zeros(3, 2)
    candidate = torch.tensor([[3.0, 4.0], [0.0, 1.0], [2.0, 0.0]])
    result = _gradient_attribution(
        reference,
        candidate,
        video_ids=["v0", "v1", "v2"],
        group_ids=["g0", "g1"],
        owner=torch.tensor([0, 0, 1]),
    )
    assert result["largest_video_gradient_discrepancies"][0]["video_id"] == "v0"
    assert result["largest_group_gradient_discrepancies"][0]["group_id"] == "g0"


def test_weak_performance_diagnostic_records_rank_changes_winners_and_ties(monkeypatch):
    groups = (
        SimpleNamespace(
            group_id="g0",
            videos=(SimpleNamespace(video_id="v0"), SimpleNamespace(video_id="v1")),
        ),
        SimpleNamespace(group_id="g1", videos=(SimpleNamespace(video_id="v2"),)),
    )
    index = SimpleNamespace(groups=groups)
    monkeypatch.setattr(
        "pmgr.diagnostics.GroupDataset", lambda *_args, **_kwargs: SimpleNamespace(index=index)
    )
    monkeypatch.setattr(
        "pmgr.diagnostics.build_retriever", lambda _config, checkpoint, device: (checkpoint, {})
    )
    score_by_model = {
        "baseline": np.array([[0.9, 0.1], [0.8, 0.1], [0.2, 0.85]], dtype=np.float32),
        "candidate": np.array([[0.95, 0.1], [0.1, 0.9], [0.2, 0.85]], dtype=np.float32),
    }

    def evaluate(model, _config, *, split, device):
        del device
        scores = score_by_model[model]
        metrics = evaluate_protocol(
            scores,
            video_ids=["v0", "v1", "v2"],
            group_ids=["g0", "g1"],
            video_to_group=[0, 0, 1],
        )
        metrics["split"] = split
        return scores, metrics

    monkeypatch.setattr("pmgr.diagnostics.evaluate_model", evaluate)
    config = {
        "paths": {"validation_index": "fixture"},
        "data": {"max_features": 4, "feature_mix_alpha": 0.8},
    }
    result = weak_performance_diagnostic(
        config,
        baseline_checkpoint="baseline",
        candidate_checkpoint="candidate",
        device=torch.device("cpu"),
    )
    baseline_g0 = result["evaluations"]["baseline"]["groups"][0]
    assert baseline_g0["candidate_queries_with_exact_group_max_tie"] == 1
    assert baseline_g0["group_max_winning_frequency"] == {"v0": 2, "v1": 0}
    assert len(result["query_rank_changes"]["T2V"]) == 2
    assert len(result["query_rank_changes"]["V2T"]) == 3
    assert result["rank_change_counts"]["V2T"]["degraded"] == 1
