from __future__ import annotations

import random

import pytest
import torch

from ocem.data.pseudoclips import PseudoClipDataset, PseudoClipError
from ocem.training.adaptation_run import (
    AdaptationRunError,
    eval_preprocess,
    run_p14t_i3d_adaptation,
)


def test_seeded_dataset_temporal_sampling_is_epoch_stable(monkeypatch) -> None:
    calls: list[int] = []

    def fake_decode(record, *, training, rng):
        calls.append(rng.randint(0, 1_000_000))
        return {"pseudo_id": record["pseudo_id"]}

    monkeypatch.setattr("ocem.data.pseudoclips.decode_pseudoclip", fake_decode)
    records = [{"pseudo_id": "a"}, {"pseudo_id": "b"}]
    dataset = PseudoClipDataset(records, training=True, seed=0, epoch=3)
    assert dataset[0] == {"pseudo_id": "a"}
    first = calls[-1]
    dataset[0]
    assert calls[-1] == first
    other_epoch = PseudoClipDataset(records, training=True, seed=0, epoch=4)
    other_epoch[0]
    assert calls[-1] != first


def test_dataset_rejects_ambiguous_randomness() -> None:
    with pytest.raises(PseudoClipError, match="either"):
        PseudoClipDataset([{"pseudo_id": "a"}], training=True, seed=0, rng=random.Random(0))


def test_eval_preprocessing_is_deterministic_center_crop() -> None:
    rgb = torch.linspace(0, 1, steps=2 * 3 * 16 * 224 * 224).reshape(2, 3, 16, 224, 224)
    first = eval_preprocess(rgb)
    second = eval_preprocess(rgb)
    assert torch.equal(first, second)
    assert first.shape == rgb.shape
    assert float(first.min()) >= -0.5
    assert float(first.max()) <= 0.5


def test_full_runner_rejects_recipe_drift_before_resources(tmp_path) -> None:
    with pytest.raises(AdaptationRunError, match="locked P14T recipe"):
        run_p14t_i3d_adaptation(
            index=tmp_path / "missing.jsonl",
            expected_index_sha256="0" * 64,
            checkpoint=tmp_path / "missing.pt",
            expected_checkpoint_sha256="0" * 64,
            implementation=tmp_path / "missing.py",
            expected_implementation_sha256="0" * 64,
            output_dir=tmp_path / "run",
            epochs=14,
        )
