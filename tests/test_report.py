from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from statistics import median

import pytest

from elsc.config import config_hash, load_config
from elsc.report import (
    TRAINING_SOURCE_PATHS,
    ReportContractError,
    _implementation_source_contract,
    compare_runs,
)
from elsc.utils import ordered_hash, sha256_file


def _write_run(
    root: Path,
    seed: int,
    ranks: tuple[list[int], list[int]],
    id_prefix: str = "",
) -> Path:
    root.mkdir(parents=True)
    dev_manifest = root / "dev.jsonl"
    dev_manifest.write_text("{}\n", encoding="utf-8")
    train_manifest = root / "train.jsonl"
    train_manifest.write_text("{}\n", encoding="utf-8")
    (root / "resolved_config.yaml").write_text(
        f"""schema_version: 1
seed: {seed}
method: baseline
upstream:
  cico_commit: pinned-upstream
data:
  train_manifest: {train_manifest}
  dev_manifest: {dev_manifest}
  dataset: ph
  caption_language: en
  feature_dim: 1024
  feature_len: 64
  max_words: 32
  combine_type: sum
  alpha: 0.9
  feature_path_mode: corrected
  sampling: upstream_uniform
  text_augmentation: cico_random_swap_v1
model:
  init_checkpoint_sha256: fixture-initialization
  backbone_frozen: false
  logit_scale_frozen: false
  adapter:
    enabled: false
    radius: 0
    hidden_dim: 256
    zero_init_output: true
  sim_header: Filip
  dual_mix: 0.5
  mix_design: balance
evaluation:
  scorer: cico_mixed_token_interaction
  metrics: upstream_cico
  full_gallery: true
  filter_by_aux_eligibility: false
train:
  epochs: 20
  per_device_batch: 512
  accumulation_steps: 1
  optimizer: adamw
  beta1: 0.9
  beta2: 0.98
  epsilon: 0.000001
  exclude_bias_and_1d_from_weight_decay: true
  core_lr: 0.00001
  adapter_lr: 0.0001
  head_lr: 0.0001
  weight_decay: 0.001
  warmup_ratio: 0.1
  schedule: cosine
  grad_clip_norm: 1.0
  precision: amp_bf16
  checkpoint_metric: mean_t2v_v2t_r1
""",
        encoding="utf-8",
    )
    checkpoint = root / "checkpoints" / "best_dev.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"selected checkpoint")
    resolved = load_config(root / "resolved_config.yaml", validate=False)
    selection = {
        "selection_split": "dev",
        "test_used_for_selection": False,
        "checkpoint": "checkpoints/best_dev.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_hash": config_hash(resolved),
        "dev_manifest_sha256": sha256_file(dev_manifest),
    }
    (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (root / "provenance.json").write_text(
        json.dumps({"initialization": {"sha256": "fixture-initialization"}}),
        encoding="utf-8",
    )
    output = root / "evaluation" / "dev"
    output.mkdir(parents=True)
    directions = {}
    ids = [f"{id_prefix}v0", f"{id_prefix}v1", f"{id_prefix}v2"]
    text_ids = [f"{id_prefix}t0", f"{id_prefix}t1", f"{id_prefix}t2"]
    for direction, values in zip(("V2T", "T2V"), ranks, strict=True):
        directions[direction] = {
            "R1": 100.0 * sum(rank < 1 for rank in values) / 3,
            "R5": 100.0 * sum(rank < 5 for rank in values) / 3,
            "R10": 100.0 * sum(rank < 10 for rank in values) / 3,
            "MedianR": 1.0 + median(values),
            "MeanR": 1.0 + sum(values) / 3,
            "cols": values,
        }
        query_ids = ids if direction == "V2T" else text_ids
        candidate_ids = text_ids if direction == "V2T" else ids
        directions.setdefault("per_query", {})[direction] = [
            {
                "query_id": query,
                "rank": rank,
                "matched_positive_id": text_ids[index] if direction == "V2T" else ids[index],
                "ranked_candidate_ids": candidate_ids,
                "official_tie_ranks": [rank],
            }
            for index, (query, rank) in enumerate(zip(query_ids, values, strict=True))
        ]
    metrics = {
        "schema_version": 1,
        "score_orientation": "video_x_text",
        "units": "percent",
        "split": "dev",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "gallery": {"videos": len(ids), "texts": len(text_ids)},
        "metric_kernel": "cico_direction_specific_singleton_tie_behavior",
        "id_hashes": {
            "videos": ordered_hash(ids),
            "texts": ordered_hash(text_ids),
            "positive_mapping": "fixture-positive-mapping",
        },
        **directions,
    }
    (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return root


def test_paired_report_uses_query_identity_and_video_groups(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([1, 0, 2], [1, 0, 2]))
    method = _write_run(tmp_path / "method", 42, ([0, 0, 2], [0, 0, 2]))
    result = compare_runs(
        [baseline], [method], split="dev", bootstrap_samples=200, bootstrap_seed=7
    )
    assert result["result_kind"] == "measured_local"
    assert result["delta"]["V2T"]["R1"]["mean"] == pytest.approx(100 / 3)
    mean_ci = result["paired_bootstrap"]["delta_percentage_points"]["MEAN"]["R1"]
    assert mean_ci["ci95"][0] <= 100 / 3 <= mean_ci["ci95"][1]


def test_paired_report_rejects_different_galleries(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]), "left-")
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]), "right-")
    with pytest.raises(ReportContractError, match="gallery ID hashes differ"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)


def test_paired_report_rejects_metrics_from_nonselected_checkpoint(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]))
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]))
    metrics_path = method / "evaluation" / "dev" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["checkpoint"]["sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ReportContractError, match="dev-selected checkpoint"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)


@pytest.mark.parametrize("missing", ["metric_kernel", "gallery"])
def test_paired_report_rejects_missing_full_gallery_contract(tmp_path: Path, missing: str):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]))
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]))
    for run in (baseline, method):
        metrics_path = run / "evaluation" / "dev" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.pop(missing)
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ReportContractError, match="metric kernel|gallery"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)


def test_paired_report_rejects_truncated_candidate_gallery(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]))
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]))
    metrics_path = method / "evaluation" / "dev" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["per_query"]["V2T"][0]["ranked_candidate_ids"].pop()
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ReportContractError, match="full gallery"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)


def test_paired_report_rejects_summary_that_disagrees_with_query_ranks(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]))
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]))
    metrics_path = method / "evaluation" / "dev" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["V2T"]["R1"] = 0.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ReportContractError, match="does not match per-query ranks"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)


def test_paired_report_rejects_different_controlled_evaluation_config(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]))
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]))
    config_path = method / "resolved_config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("alpha: 0.9", "alpha: 0.8"),
        encoding="utf-8",
    )
    config = load_config(config_path, validate=False)
    selection_path = method / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["config_hash"] = config_hash(config)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ReportContractError, match="evaluation contracts differ"):
        compare_runs([baseline], [method], split="dev", bootstrap_samples=100)


def test_training_source_contract_uses_recorded_commit_blobs(monkeypatch, tmp_path: Path):
    output = "\n".join(
        f"100644 blob {index:040x}\t{path}" for index, path in enumerate(TRAINING_SOURCE_PATHS, 1)
    )
    monkeypatch.setattr(
        "elsc.report.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    provenance = {
        "implementation": {
            "status": "ready",
            "tracked_worktree_dirty": False,
            "root": str(tmp_path),
            "commit": "recorded-commit",
        }
    }
    result = _implementation_source_contract(provenance, tmp_path / "run")
    assert len(result["git_blob_ids"]) == len(TRAINING_SOURCE_PATHS)
    assert len(result["source_hash"]) == 64

    provenance["implementation"]["tracked_worktree_dirty"] = True
    with pytest.raises(ReportContractError, match="dirty tracked sources"):
        _implementation_source_contract(provenance, tmp_path / "run")


def test_report_can_require_matched_optimization_budget(tmp_path: Path):
    baseline = _write_run(tmp_path / "base", 42, ([0, 0, 0], [0, 0, 0]))
    method = _write_run(tmp_path / "method", 42, ([0, 0, 0], [0, 0, 0]))
    config_path = method / "resolved_config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("epochs: 20", "epochs: 10"),
        encoding="utf-8",
    )
    config = load_config(config_path, validate=False)
    selection_path = method / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["config_hash"] = config_hash(config)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    compare_runs([baseline], [method], split="dev", bootstrap_samples=100)
    with pytest.raises(ReportContractError, match="optimization budget contracts differ"):
        compare_runs(
            [baseline],
            [method],
            split="dev",
            bootstrap_samples=100,
            require_matched_training_budget=True,
        )
