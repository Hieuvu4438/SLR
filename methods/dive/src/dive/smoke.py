from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .config import config_hash
from .data.fixtures import build_synthetic_fixture
from .data.relations import build_pair_relations
from .eval.metrics import evaluate_retrieval
from .models.evidence import EvidenceEncoder, clone_reference_and_student, state_hash
from .models.scoring import compose_score, evidence_score_block
from .training.state import load_training_checkpoint, save_training_checkpoint
from .training.step import run_student_step


class _FixturePoseEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, pose: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        features = self.projection(pose)
        shape = features.shape
        features = self.batch_norm(features.reshape(-1, shape[-1])).reshape(shape)
        clips = []
        for batch, sample_grid in enumerate(grid):
            clips.append(
                torch.stack([features[batch, left:right].mean(0) for left, right in sample_grid])
            )
        return torch.stack(clips)


@dataclass(frozen=True)
class _SmokeContrast:
    sample_i: str
    sample_j: str
    unit_i: int
    unit_j: int
    q_i: tuple[float, ...] | None
    q_j: tuple[float, ...] | None
    h: float | None
    g: float


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_fixture_smoke(
    config: Mapping[str, Any], output_dir: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Run one deterministic CPU DIVE step and checkpoint reload; never a benchmark."""
    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    fixture = build_synthetic_fixture(seed)
    count = len(fixture.sample_ids)
    pose = torch.randn(count, 5, 3)
    rgb = torch.randn(count, 2, 5)
    grid = torch.tensor([[[0, 3], [2, 5]]] * count, dtype=torch.int64)
    video_mask = torch.ones(count, 2, dtype=torch.bool)
    output_dim = int(config["evidence"]["output_dim"])
    text_units = torch.nn.functional.normalize(torch.randn(count, 2, output_dim), dim=-1)
    text_mask = torch.ones(count, 2, dtype=torch.bool)
    winner = EvidenceEncoder(
        _FixturePoseEncoder(4),
        rgb_dim=5,
        pose_dim=4,
        hidden_dim=int(config["evidence"]["hidden_dim"]),
        output_dim=output_dim,
        normalize_epsilon=float(config["evidence"]["normalize_epsilon"]),
    )
    pair = clone_reference_and_student(winner)
    with torch.no_grad():
        reference_u = pair.reference(pose, rgb, grid, video_mask)
    positives_by_video = {
        video_id: {text_id}
        for video_id, text_id in zip(fixture.video_ids, fixture.text_ids, strict=True)
    }
    relations = build_pair_relations(
        fixture.video_ids,
        fixture.text_ids,
        positives_by_video,
        fixture.excluded_negatives,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in pair.student.parameters() if parameter.requires_grad],
        lr=float(config["train"]["lr_projector"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    contrasts = (
        _SmokeContrast(
            fixture.sample_ids[0], fixture.sample_ids[1], 0, 0, (0.7, 0.3), (0.4, 0.6), 1.0, 0.8
        ),
        _SmokeContrast(
            fixture.sample_ids[2], fixture.sample_ids[3], -1, -1, None, None, None, 0.0
        ),
    )
    sample_rows = {sample: index for index, sample in enumerate(fixture.sample_ids)}
    metrics = run_student_step(
        pair.student,
        optimizer,
        pose=pose,
        rgb_local=rgb,
        grid=grid,
        video_mask=video_mask,
        text_units=text_units,
        text_mask=text_mask,
        baseline_scores=fixture.baseline_scores,
        reference_evidence=reference_u,
        positives=relations.positives,
        candidates=relations.candidates,
        sample_to_video_row=sample_rows,
        sample_to_text_column=sample_rows,
        sampled_contrasts=contrasts,
        gamma_train=float(config["train"]["gamma_train"]),
        tau_alignment=float(config["evidence"]["tau_alignment"]),
        tau_retrieval=float(config["loss"]["tau_retrieval"]),
        tau_pair=float(config["loss"]["tau_pair"]),
        tau_local_margin=float(config["loss"]["tau_local_margin"]),
        local_margin_alpha=float(config["loss"]["local_margin_alpha"]),
        lambda_pair=float(config["loss"]["lambda_pair"]),
        lambda_local=float(config["loss"]["lambda_local"]),
        grad_clip_norm=float(config["train"]["grad_clip_norm"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "fixture_student.pt"
    fingerprints = {"data": "synthetic_fixture_v1", "bank": "fixture_bank_v1", "reference": pair.initial_state_hash}
    digest = config_hash(config)
    checksum = save_training_checkpoint(
        checkpoint_path,
        model=pair.student,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=0,
        global_step=1,
        best_dev={"fixture_only": True},
        sampler_state={"plan_hash": "fixture_plan_v1", "cursor": 1},
        fingerprints=fingerprints,
        config_hash=digest,
        git_revision=_git_revision(Path(repository_root)),
        optimizer_manifest={"fixture": True, "trainable_parameters": len(optimizer.param_groups[0]["params"])},
    )
    reloaded = copy.deepcopy(pair.student)
    reloaded_optimizer = torch.optim.AdamW(
        [parameter for parameter in reloaded.parameters() if parameter.requires_grad],
        lr=float(config["train"]["lr_projector"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    resume = load_training_checkpoint(
        checkpoint_path,
        model=reloaded,
        optimizer=reloaded_optimizer,
        scheduler=None,
        scaler=None,
        expected_config_hash=digest,
        expected_fingerprints=fingerprints,
        restore_rng=False,
    )
    with torch.no_grad():
        student_u = reloaded(pose, rgb, grid, video_mask)
        student_score, pair_valid = evidence_score_block(
            student_u, text_units, video_mask, text_mask, float(config["evidence"]["tau_alignment"])
        )
        reference_score, _ = evidence_score_block(
            reference_u, text_units, video_mask, text_mask, float(config["evidence"]["tau_alignment"])
        )
        scores = compose_score(
            fixture.baseline_scores,
            student_score,
            reference_score,
            pair_valid,
            float(config["train"]["gamma_train"]),
        )
    retrieval = evaluate_retrieval(
        scores.numpy(), fixture.video_ids, fixture.text_ids, fixture.relevance, topk=(1, 2)
    )
    report = {
        "schema_version": "fixture_smoke.v1",
        "fixture_only": True,
        "benchmark_claim_allowed": False,
        "device": "cpu",
        "precision": "float32",
        "config_hash": digest,
        "checkpoint_sha256": checksum,
        "checkpoint_round_trip": state_hash(reloaded) == state_hash(pair.student),
        "resume_global_step": resume.global_step,
        "step_metrics": asdict(metrics),
        "retrieval_contract_metrics": retrieval.to_dict(),
    }
    report_path = output / "smoke_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report
