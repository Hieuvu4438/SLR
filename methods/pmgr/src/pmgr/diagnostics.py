from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pmgr.config import load_config
from pmgr.data.group_dataset import GroupCollator, GroupDataset
from pmgr.data.group_sampler import GroupBatchSampler
from pmgr.evaluate import evaluate_model
from pmgr.losses import objective_loss, whole_group_max
from pmgr.metrics import group_max_scores
from pmgr.model import build_retriever, load_tokenizer
from pmgr.scoring import mixed_pair_scores
from slr_common.utils import atomic_json_dump, stable_seed


def _gradient(loss: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(loss, scores, retain_graph=False)[0]


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    left, right = reference.flatten(), candidate.flatten()
    left_norm, right_norm = left.norm(), right.norm()
    cosine = F.cosine_similarity(left[None], right[None]).item()
    return {
        "direction_cosine": float(cosine),
        "norm_ratio_to_group_ce": float(right_norm / left_norm.clamp_min(1e-12)),
        "group_ce_norm": float(left_norm),
        "candidate_norm": float(right_norm),
    }


def population_diagnostic(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    device: torch.device,
    groups: int,
    draws: int,
) -> dict[str, Any]:
    if groups < 2 or draws < 1:
        raise ValueError("population diagnostic requires at least two groups and one draw")
    dataset = GroupDataset(
        config["paths"]["train_index"], feature_len=int(config["data"]["max_features"]),
        alpha=float(config["data"]["feature_mix_alpha"]),
    )
    sampler = GroupBatchSampler(len(dataset), groups, seed=int(config["seed"]), drop_last=True)
    indexes = next(iter(sampler))
    tokenizer = load_tokenizer(config)
    collator = GroupCollator(
        tokenizer, int(config["data"]["max_text_tokens"]),
        dataset_group_count=dataset.index.group_count, dataset_video_count=dataset.index.video_count,
        seed=int(config["seed"]), augment=bool(config["data"]["text_augmentation"]["enabled"]),
        augmentation_probability=float(config["data"]["text_augmentation"]["probability"]),
    )
    collator.set_position(0, 0)
    batch = collator([dataset[index] for index in indexes])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    model, _ = build_retriever(config, checkpoint=checkpoint, device=device)
    model.eval()
    with torch.no_grad():
        encoded = model.encode_pmgr_batch(batch)
        scoring = config["scoring"]
        q, _, _ = mixed_pair_scores(
            encoded.video_hidden, encoded.text_hidden, encoded.aug_hidden,
            encoded.video_valid, encoded.text_valid, encoded.aug_valid,
            omega=float(scoring["dual_mix"]), sigma=float(scoring["inner_temperature"]),
            normalize_eps=float(scoring["normalize_eps"]), mask_policy=scoring["mask_policy"],
        )
    owner = batch["video_to_group"]
    leaf = q.detach().requires_grad_(True)
    common = dict(
        video_to_group=owner, logit_scale=model.logit_scale,
        dataset_group_count=batch["dataset_group_count"],
        dataset_video_count=batch["dataset_video_count"],
    )
    group_output = objective_loss("group_ce", leaf, **common)
    group_gradient = _gradient(group_output["loss"], leaf)
    comparisons: dict[str, Any] = {}
    for mode in ("all_uniform_ce", "all_set_ce"):
        candidate = q.detach().requires_grad_(True)
        output = objective_loss(mode, candidate, **common)
        comparisons[mode] = _comparison(group_gradient, _gradient(output["loss"], candidate))
    group_scores = whole_group_max(q, owner)
    draw_reports = []
    for draw in range(draws):
        positions = []
        for group in range(groups):
            members = torch.nonzero(owner == group, as_tuple=False).flatten()
            generator = torch.Generator(device="cpu").manual_seed(
                stable_seed(config["seed"], draw, batch["group_ids"][group], "diagnostic_representative")
            )
            offset = int(torch.randint(len(members), (1,), generator=generator))
            positions.append(int(members[offset]))
        selected = torch.tensor(positions, device=device)
        representative = q.index_select(0, selected).detach().requires_grad_(True)
        rep_owner = torch.arange(groups, device=device)
        output = objective_loss(
            "single_mixed_ce", representative, rep_owner, model.logit_scale,
            batch["dataset_group_count"], batch["dataset_video_count"],
        )
        rep_gradient = _gradient(output["loss"], representative)
        embedded = torch.zeros_like(group_gradient)
        embedded.index_add_(0, selected, rep_gradient)
        full_order = torch.argsort(group_scores, dim=1, descending=True, stable=True)
        rep_order = torch.argsort(representative.T, dim=1, descending=True, stable=True)
        draw_reports.append(
            {
                "draw": draw,
                "selected_video_ids": [batch["video_ids"][position] for position in positions],
                "queries_with_changed_candidate_order": int((full_order != rep_order).any(dim=1).sum()),
                **_comparison(group_gradient, embedded),
            }
        )
    return {
        "schema_version": 1,
        "mode": "population",
        "checkpoint": str(checkpoint),
        "group_ids": batch["group_ids"],
        "full_group_sizes": batch["full_group_sizes"].cpu().tolist(),
        "loaded_videos": len(batch["video_ids"]),
        "group_ce_loss": float(group_output["loss"].detach()),
        "all_performance_controls": comparisons,
        "representative_draws": draw_reports,
        "interpretation": "gradient/order discrepancy is diagnostic only, not evidence of improvement",
    }


def _direction_margins(scores, targets):
    rows = torch.as_tensor(scores, dtype=torch.float64)
    target = torch.as_tensor(targets, dtype=torch.long)
    positive = rows.gather(1, target[:, None]).squeeze(1)
    negative = rows.clone()
    negative[torch.arange(len(rows)), target] = -torch.inf
    return (positive - negative.max(dim=1).values).numpy()


def weak_performance_diagnostic(
    config: dict[str, Any],
    *,
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    index = GroupDataset(
        config["paths"]["validation_index"],
        feature_len=int(config["data"]["max_features"]),
        alpha=float(config["data"]["feature_mix_alpha"]),
    ).index
    group_ids = [group.group_id for group in index.groups]
    video_ids = [video.video_id for group in index.groups for video in group.videos]
    owner = [group for group, item in enumerate(index.groups) for _ in item.videos]
    evaluations = {}
    for label, checkpoint in (
        ("baseline", baseline_checkpoint), ("candidate", candidate_checkpoint)
    ):
        model, _ = build_retriever(config, checkpoint=checkpoint, device=device)
        scores, metrics = evaluate_model(model, config, split="validation", device=device)
        grouped = group_max_scores(scores, owner, len(group_ids))
        positive = scores[range(len(video_ids)), owner]
        v2t_margins = _direction_margins(scores, owner)
        t2v_margins = _direction_margins(grouped, range(len(group_ids)))
        per_group = []
        for group, record in enumerate(index.groups):
            members = [position for position, value in enumerate(owner) if value == group]
            ranks = [metrics["V2T"]["ranks"][position] for position in members]
            member_scores = positive[members]
            per_group.append(
                {
                    "group_id": record.group_id,
                    "size": len(members),
                    "t2v_rank": metrics["T2V"]["ranks"][group],
                    "v2t_rank_min": min(ranks),
                    "v2t_rank_mean": float(sum(ranks) / len(ranks)),
                    "v2t_rank_max": max(ranks),
                    "positive_score_spread": float(member_scores.max() - member_scores.min()),
                    "t2v_margin": float(t2v_margins[group]),
                    "v2t_margin_min": float(v2t_margins[members].min()),
                    "v2t_margin_mean": float(v2t_margins[members].mean()),
                }
            )
        evaluations[label] = {"checkpoint": str(checkpoint), "metrics": metrics, "groups": per_group}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_groups = evaluations["baseline"]["groups"]
    candidate_groups = evaluations["candidate"]["groups"]
    concentration_failures = []
    for before, after in zip(baseline_groups, candidate_groups, strict=True):
        if after["t2v_rank"] < before["t2v_rank"] and after["v2t_rank_max"] > before["v2t_rank_max"]:
            concentration_failures.append(
                {
                    "group_id": before["group_id"],
                    "t2v_rank_before": before["t2v_rank"],
                    "t2v_rank_after": after["t2v_rank"],
                    "worst_v2t_before": before["v2t_rank_max"],
                    "worst_v2t_after": after["v2t_rank_max"],
                }
            )
    size_summary = {}
    for size in sorted({len(group.videos) for group in index.groups}):
        positions = [position for position, group in enumerate(index.groups) if len(group.videos) == size]
        size_summary[str(size)] = {
            "groups": len(positions),
            "baseline_mean_worst_v2t_rank": float(
                sum(baseline_groups[position]["v2t_rank_max"] for position in positions) / len(positions)
            ),
            "candidate_mean_worst_v2t_rank": float(
                sum(candidate_groups[position]["v2t_rank_max"] for position in positions) / len(positions)
            ),
        }
    return {
        "schema_version": 1,
        "mode": "weak_performance",
        "split": "validation",
        "test_accessed": False,
        "evaluations": evaluations,
        "groups_with_t2v_improvement_and_worst_v2t_degradation": concentration_failures,
        "group_size_summary": size_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PMGR mechanism diagnostics without weight updates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("population", "weak_performance"), default="population")
    parser.add_argument("--checkpoint")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    config = load_config(args.config, mode="train")
    if args.mode == "population":
        checkpoint = args.checkpoint or config["paths"]["initialization_checkpoint"]
        result = population_diagnostic(
            config, checkpoint=checkpoint, device=torch.device(args.device),
            groups=args.groups, draws=args.draws,
        )
        output = Path(args.output_dir) / "population.json"
    else:
        if not args.candidate_checkpoint:
            raise ValueError("weak_performance requires --candidate-checkpoint")
        result = weak_performance_diagnostic(
            config,
            baseline_checkpoint=args.baseline_checkpoint or config["paths"]["initialization_checkpoint"],
            candidate_checkpoint=args.candidate_checkpoint,
            device=torch.device(args.device),
        )
        output = Path(args.output_dir) / "weak_performance.json"
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
