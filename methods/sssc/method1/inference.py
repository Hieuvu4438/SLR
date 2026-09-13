from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .audit import audit_resources
from .baseline import directional_scores_blocked
from .checkpoints import atomic_torch_save
from .config import Method1Config
from .data import _load_records
from .evaluation import evaluate_grouped_retrieval
from .model_factory import build_upret_model, load_exact_student_state
from .reference_pipeline import _video_batch
from .schemas import GroupRecord, TextRecord, VideoRecord
from .token_spans import tokenize_with_spans
from .upstream import create_upret_tokenizer
from .utils import atomic_json_dump, sha256_file


@torch.no_grad()
def _encode_inference_pool(
    model: torch.nn.Module,
    config: Method1Config,
    *,
    split: str,
    tokenizer: Any,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Any], list[Any]]:
    root = Path(config.data.manifest_dir)
    videos = sorted(
        (record for record in _load_records(root / "videos.jsonl", VideoRecord) if record.split == split),
        key=lambda record: record.official_order,
    )
    texts = sorted(
        (record for record in _load_records(root / "texts.jsonl", TextRecord) if record.split == split),
        key=lambda record: record.official_order,
    )
    groups = sorted(
        (
            record
            for record in _load_records(root / "groups.jsonl", GroupRecord)
            if record.group_uid.startswith(f"{config.data.dataset}:{split}:")
        ),
        key=lambda record: record.official_order,
    )
    if [record.text_uid for record in groups] != [record.text_uid for record in texts]:
        raise RuntimeError("evaluation text and group order differ")
    batch_size = config.evaluation.encode_batch_size
    video_tokens: list[torch.Tensor] = []
    video_masks: list[torch.Tensor] = []
    for start in range(0, len(videos), batch_size):
        current = videos[start : start + batch_size]
        batch, _ = _video_batch(config, current)
        batch = {key: value.to(device) for key, value in batch.items()}
        mask, tokens, _ = model.get_video_feat(
            batch["video_features"],
            batch["video_ignore_raw"],
            shaped=True,
            video_frame=1,
            get_hidden=True,
        )
        video_tokens.append(tokens.float().cpu())
        video_masks.append(mask.bool().cpu())
    text_tokens: list[torch.Tensor] = []
    text_masks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        current = texts[start : start + batch_size]
        encoded = [
            tokenize_with_spans(
                text.raw_text,
                text_uid=text.text_uid,
                tokenizer=tokenizer,
                max_positions=config.data.text_max_positions,
            )
            for text in current
        ]
        ids = torch.tensor([item.input_ids for item in encoded], dtype=torch.long, device=device)
        valid = torch.tensor([item.text_valid for item in encoded], dtype=torch.bool, device=device)
        returned_valid, tokens, _ = model.get_text_feat(
            ids, torch.zeros_like(ids), valid, shaped=False, get_hidden=True
        )
        if not torch.equal(returned_valid.bool(), valid):
            raise RuntimeError("inference tokenizer and text encoder masks differ")
        text_tokens.append(tokens.float().cpu())
        text_masks.append(returned_valid.bool().cpu())
    return (
        torch.cat(video_tokens),
        torch.cat(video_masks),
        torch.cat(text_tokens),
        torch.cat(text_masks),
        videos,
        groups,
    )


def evaluate_loaded_student(
    config: Method1Config,
    model: torch.nn.Module,
    *,
    split: str,
    tokenizer: Any,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise ValueError("evaluation split must be dev or test")
    was_training = model.training
    model.eval()
    video, video_ignore, text, text_valid, videos, groups = _encode_inference_pool(
        model,
        config,
        split=split,
        tokenizer=tokenizer,
        device=device,
    )
    with torch.no_grad():
        video_score, text_score = directional_scores_blocked(
            model,
            video.to(device),
            video_ignore.to(device),
            text.to(device),
            text_valid.to(device),
            text_aug_raw=text.to(device),
            text_aug_valid=text_valid.to(device),
            temperature=config.model.inner_similarity_temperature,
            video_block=config.training.train_video_pair_block,
            text_block=config.training.train_text_pair_block,
        )
    beta = config.model.dual_mix
    scores = (beta * video_score + (1.0 - beta) * text_score).float().cpu().numpy()
    group_to_index = {group.group_uid: index for index, group in enumerate(groups)}
    video_group_indexes = [group_to_index[record.group_uid] for record in videos]
    evaluated = evaluate_grouped_retrieval(
        scores,
        video_group_indexes=video_group_indexes,
        top_k=config.evaluation.report_top_k,
    )
    grouped_scores = evaluated.pop("grouped_t2v_scores")
    video_ids = [record.video_uid for record in videos]
    group_ids = [record.group_uid for record in groups]
    evaluated["top_candidate_ids"] = {
        "V2T": [
            [group_ids[index] for index in row]
            for row in evaluated["top_candidate_indexes"]["V2T"]
        ],
        "T2V": [
            [group_ids[index] for index in row]
            for row in evaluated["top_candidate_indexes"]["T2V"]
        ],
    }
    evaluated["query_ids"] = {"V2T": video_ids, "T2V": group_ids}
    evaluated["query_group_ids"] = {
        "V2T": [record.group_uid for record in videos],
        "T2V": group_ids,
    }
    report = {
        **evaluated,
        "status": "complete",
        "dataset": config.data.dataset,
        "split": split,
        "config_sha256": config.digest,
        "manifest_hashes": {
            name: sha256_file(Path(config.data.manifest_dir) / name)
            for name in ("texts.jsonl", "videos.jsonl", "groups.jsonl")
        },
        "precision": "float32",
        "scorer": "upret_main_corrected_v1_mixed_directional",
        "group_score_shape": list(grouped_scores.shape),
    }
    if was_training:
        model.train()
    return report


def evaluate_checkpoint(
    config: Method1Config,
    checkpoint: str | Path,
    *,
    split: str,
    upret_root: str | Path = "third_party/UPRet",
    device: str | torch.device = "cpu",
    persist: bool = True,
) -> dict[str, Any]:
    audit_resources(config, "inference")
    model, model_report = build_upret_model(config, upret_root=upret_root)
    checkpoint_payload = load_exact_student_state(model, checkpoint)
    model = model.to(device).eval()
    tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
    report = evaluate_loaded_student(
        config, model, split=split, tokenizer=tokenizer, device=device
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    report.update(
        {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_arm": checkpoint_payload.get("arm"),
            "model_architecture": model_report["architecture"],
        }
    )
    if persist:
        output = (
            Path(config.output.root)
            / "evaluation"
            / f"{split}_{checkpoint_sha256[:12]}.json"
        )
        atomic_json_dump(report, output)
        report["metrics_path"] = str(output.resolve())
    return report


def export_student(
    config: Method1Config,
    checkpoint: str | Path,
    output: str | Path,
    *,
    upret_root: str | Path = "third_party/UPRet",
) -> dict[str, Any]:
    audit_resources(config, "inference")
    model, model_report = build_upret_model(config, upret_root=upret_root)
    source = load_exact_student_state(model, checkpoint)
    resolved_config = asdict(config)
    resolved_config.pop("source_path", None)
    artifact = {
        "checkpoint_schema_version": 1,
        "artifact_type": "method1_student_inference",
        "method_version": config.method_version,
        "baseline_version": config.baseline_version,
        "upstream_commit": config.upstream_commit,
        "student_state_dict": model.state_dict(),
        "resolved_config": resolved_config,
        "config_sha256": config.digest,
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_arm": source.get("arm"),
        "architecture": model_report["architecture"],
        "training_only_dependencies": [],
    }
    atomic_torch_save(artifact, output)
    return {
        "schema_version": 1,
        "status": "complete",
        "output": str(Path(output).resolve()),
        "sha256": sha256_file(output),
        "source_checkpoint_sha256": artifact["source_checkpoint_sha256"],
        "student_parameter_keys": len(artifact["student_state_dict"]),
        "training_only_dependencies": [],
    }
