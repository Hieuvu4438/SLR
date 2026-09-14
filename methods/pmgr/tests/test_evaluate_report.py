from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pmgr.evaluate import _encode_gallery
from pmgr.report import build_phase_b_report


class _Gallery:
    def __init__(self):
        self.items = [
            {
                "group_id": "g0",
                "canonical_text": "zero",
                "features": torch.arange(24, dtype=torch.float32).reshape(2, 4, 3),
                "valid": torch.ones(2, 4, dtype=torch.bool),
                "video_ids": ["v0", "v1"],
            },
            {
                "group_id": "g1",
                "canonical_text": "one",
                "features": torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
                "valid": torch.ones(1, 4, dtype=torch.bool),
                "video_ids": ["v2"],
            },
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _Model:
    def __init__(self):
        self.video_batch_sizes = []
        self.text_batch_sizes = []

    def eval(self):
        return self

    def encode_video(self, features, padding_mask):
        self.video_batch_sizes.append(len(features))
        hidden = torch.zeros(len(features), padding_mask.shape[1], 2)
        return hidden, padding_mask.eq(0)

    def encode_text(self, input_ids, segment_ids, valid):
        del segment_ids
        self.text_batch_sizes.append(len(input_ids))
        return torch.zeros(len(input_ids), input_ids.shape[1], 2), valid.bool()


def test_full_gallery_encoder_retains_last_partial_batches(monkeypatch):
    def encode(_text, _tokenizer, max_words):
        return (
            torch.ones(max_words, dtype=torch.long),
            torch.zeros(max_words, dtype=torch.long),
            torch.ones(max_words, dtype=torch.long),
        )

    monkeypatch.setattr("pmgr.evaluate.encode_cico_text", encode)
    config = {
        "data": {"max_text_tokens": 3},
        "validation": {"video_encoder_batch": 2, "text_encoder_batch": 1},
    }
    model = _Model()
    result = _encode_gallery(model, _Gallery(), object(), config, torch.device("cpu"))
    video, _, text, _, video_ids, group_ids, owner = result
    assert (len(video), len(text)) == (3, 2)
    assert video_ids == ["v0", "v1", "v2"]
    assert group_ids == ["g0", "g1"]
    assert owner == [0, 0, 1]
    assert model.video_batch_sizes == [2, 1]
    assert model.text_batch_sizes == [1, 1]


def _write_run(path: Path, mode: str, r1: float) -> None:
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "training_complete",
                "checkpoint": {"path": str(path / "best.pt")},
            }
        )
    )
    (path / "resolved_config.json").write_text(json.dumps({"loss": {"mode": mode}}))
    direction = {"R1": r1, "R5": r1 + 5, "R10": r1 + 10, "MedianR": 1, "MeanR": 2}
    (path / "best_dev_metrics.json").write_text(
        json.dumps({"epoch": 2, "metrics": {"split": "validation", "T2V": direction, "V2T": direction}})
    )
    (path / "train.jsonl").write_text(
        json.dumps(
            {
                "loaded_videos": 8,
                "candidate_pairs": 32,
                "seconds_per_update": 0.5,
            }
        )
        + "\n"
        + json.dumps({"event": "epoch_complete", "effective_step": 1})
        + "\n"
    )


def test_phase_b_report_uses_matched_controls_and_ignores_event_rows(tmp_path):
    modes = {
        "C0": "legacy_cico",
        "C1": "single_mixed_ce",
        "C2": "all_uniform_ce",
        "C3": "all_set_ce",
        "C3_population": "all_set_ce_population_weighted",
        "C4": "group_ce",
    }
    r1 = {"C0": 60.0, "C1": 61.0, "C2": 62.0, "C3": 63.0, "C3_population": 62.5, "C4": 63.4}
    runs = {}
    for arm, mode in modes.items():
        runs[arm] = tmp_path / arm
        _write_run(runs[arm], mode, r1[arm])
    report = build_phase_b_report(runs)
    assert report["strongest_equal_input_positive_control"] == "C3"
    assert report["c4_delta_mean_bidirectional_r1_points"] == pytest.approx(0.4)
    assert report["status"] == "population_hypothesis_no_go"
    assert report["research_supported"] is False
    assert report["arms"]["C4"]["exposures"]["effective_steps"] == 1
