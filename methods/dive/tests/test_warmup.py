from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from dive.config import config_hash, load_config
from dive.models.evidence import EvidenceEncoder, state_hash
from dive.training.optimizer import build_evidence_optimizer
from dive.training.warmup import (
    ChunkedWarmupGallery,
    WarmupBatch,
    WarmupContractError,
    WarmupGallery,
    WarmupTextGalleryBatch,
    WarmupVideoGalleryBatch,
    evaluate_chunked_warmup_gallery,
    evaluate_warmup_gallery,
    run_evidence_warmup,
    run_warmup_step,
)


HERE = Path(__file__).resolve().parents[1]


class PointwisePose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, pose: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return self.projection(pose[:, : grid.shape[1]])


def _model() -> EvidenceEncoder:
    torch.manual_seed(101)
    return EvidenceEncoder(PointwisePose(), rgb_dim=5, pose_dim=4, hidden_dim=12, output_dim=8)


def _batch(model: EvidenceEncoder) -> WarmupBatch:
    generator = torch.Generator().manual_seed(103)
    pose = torch.randn(2, 2, 3, generator=generator)
    rgb = torch.randn(2, 2, 5, generator=generator)
    grid = torch.tensor([[[0, 1], [1, 2]], [[0, 1], [1, 2]]], dtype=torch.int64)
    video_mask = torch.ones(2, 2, dtype=torch.bool)
    with torch.no_grad():
        text = model(pose, rgb, grid, video_mask).mean(dim=1)[:, None, :]
        text = torch.nn.functional.normalize(text, dim=-1)
    return WarmupBatch(
        pose=pose,
        rgb_local=rgb,
        grid=grid,
        video_mask=video_mask,
        text_units=text,
        text_mask=torch.ones(2, 1, dtype=torch.bool),
        positives=torch.eye(2, dtype=torch.bool),
        candidates=torch.ones(2, 2, dtype=torch.bool),
    )


def test_warmup_step_updates_evidence_but_not_frozen_inputs():
    model = _model()
    batch = _batch(model)
    pose = batch.pose.clone().requires_grad_(True)
    text = batch.text_units.clone().requires_grad_(True)
    batch = WarmupBatch(**{**batch.__dict__, "pose": pose, "text_units": text})
    bundle = build_evidence_optimizer(model)
    before = state_hash(model)
    loss, gradient_norm = run_warmup_step(
        model,
        bundle.optimizer,
        batch,
        tau_alignment=0.07,
        tau_retrieval=0.07,
        grad_clip_norm=1.0,
    )
    assert loss > 0 and gradient_norm > 0
    assert state_hash(model) != before
    assert pose.grad is None and text.grad is None


def test_warmup_step_rejects_excluded_positive_relation():
    model = _model()
    batch = _batch(model)
    candidates = batch.candidates.clone()
    candidates[0, 0] = False
    invalid = WarmupBatch(**{**batch.__dict__, "candidates": candidates})
    with pytest.raises(WarmupContractError, match="positives must remain"):
        run_warmup_step(
            model,
            build_evidence_optimizer(model).optimizer,
            invalid,
            tau_alignment=0.07,
            tau_retrieval=0.07,
            grad_clip_norm=1.0,
        )


def test_chunked_gallery_matches_dense_full_gallery_exactly():
    model = _model()
    batch = _batch(model)
    relevance = {"v0": ("t0",), "v1": ("t1",)}
    dense = WarmupGallery(
        batch=batch,
        video_ids=("v0", "v1"),
        text_ids=("t0", "t1"),
        video_to_text_positives=relevance,
    )

    def video_batches():
        for index, video_id in enumerate(dense.video_ids):
            yield WarmupVideoGalleryBatch(
                pose=batch.pose[index : index + 1],
                rgb_local=batch.rgb_local[index : index + 1],
                grid=batch.grid[index : index + 1],
                video_mask=batch.video_mask[index : index + 1],
                video_ids=(video_id,),
            )

    def text_batches():
        for index, text_id in enumerate(dense.text_ids):
            yield WarmupTextGalleryBatch(
                text_units=batch.text_units[index : index + 1],
                text_mask=batch.text_mask[index : index + 1],
                text_ids=(text_id,),
            )

    chunked = ChunkedWarmupGallery(
        video_batches=video_batches,
        text_batches=text_batches,
        video_ids=dense.video_ids,
        text_ids=dense.text_ids,
        video_to_text_positives=relevance,
    )
    expected = evaluate_warmup_gallery(model, dense, tau_alignment=0.07)
    actual = evaluate_chunked_warmup_gallery(model, chunked, tau_alignment=0.07)
    assert actual.to_dict() == expected.to_dict()


def test_warmup_runner_selects_earliest_dev_tie_and_exports_reference(tmp_path):
    model = _model()
    batch = _batch(model)
    gallery = WarmupGallery(
        batch=batch,
        video_ids=("v0", "v1"),
        text_ids=("t0", "t1"),
        video_to_text_positives={"v0": ("t0",), "v1": ("t1",)},
    )
    config = load_config(HERE / "configs" / "fixture.yaml")
    config["train"]["warmup_epochs"] = 2
    result = run_evidence_warmup(
        model,
        lambda _epoch: (batch,),
        gallery,
        config=config,
        steps_per_epoch=1,
        output_dir=tmp_path,
        config_hash=config_hash(config),
        fingerprints={
            "baseline": "fixture",
            "data": "fixture",
            "units": "fixture",
            "grid": "fixture",
        },
        git_revision="fixture_revision",
    )
    assert result.winner.epoch == 1
    assert result.winner.endpoint == 1.0
    assert Path(result.reference_path).is_file()
    assert Path(result.reference_path + ".sha256").read_text(encoding="ascii").strip() == (
        result.reference_checksum
    )
    selection = json.loads(Path(result.selection_path).read_text(encoding="utf-8"))
    assert selection["selection_split"] == "dev"
    assert selection["teacher_only"] is True
    assert selection["winner"]["epoch"] == 1
    assert selection["fingerprints"]["dev_gallery"]
    reference = torch.load(result.reference_path, map_location="cpu", weights_only=False)
    assert reference["state_hash"] == result.reference_state_hash == state_hash(model)
    assert reference["git_revision"] == "fixture_revision"
    assert reference["resolved_config"]["run"]["profile"] == "fixture"
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_warmup_resume_restores_scheduler_history_and_exact_reference(tmp_path):
    config = load_config(HERE / "configs" / "fixture.yaml")
    config["train"]["warmup_epochs"] = 2
    fingerprints = {
        "baseline": "fixture",
        "data": "fixture",
        "units": "fixture",
        "grid": "fixture",
    }
    output = tmp_path / "resume"
    first_model = _model()
    first_batch = _batch(first_model)
    gallery = WarmupGallery(
        batch=first_batch,
        video_ids=("v0", "v1"),
        text_ids=("t0", "t1"),
        video_to_text_positives={"v0": ("t0",), "v1": ("t1",)},
    )
    expected = run_evidence_warmup(
        first_model,
        lambda _epoch: (first_batch,),
        gallery,
        config=config,
        steps_per_epoch=1,
        output_dir=output,
        config_hash=config_hash(config),
        fingerprints=fingerprints,
        git_revision="fixture_revision",
    )
    for name in (
        "warmup_epoch_002.pt",
        "warmup_epoch_002.pt.sha256",
        "reference.pt",
        "reference.pt.sha256",
        "selection.json",
    ):
        (output / name).unlink()

    resumed_model = _model()
    resumed_batch = _batch(resumed_model)
    resumed_gallery = WarmupGallery(
        batch=resumed_batch,
        video_ids=gallery.video_ids,
        text_ids=gallery.text_ids,
        video_to_text_positives=gallery.video_to_text_positives,
    )
    actual = run_evidence_warmup(
        resumed_model,
        lambda _epoch: (resumed_batch,),
        resumed_gallery,
        config=config,
        steps_per_epoch=1,
        output_dir=output,
        config_hash=config_hash(config),
        fingerprints=fingerprints,
        git_revision="fixture_revision",
        resume=True,
    )
    assert actual.reference_state_hash == expected.reference_state_hash
    assert actual.candidates == expected.candidates
