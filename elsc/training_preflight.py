from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from elsc.config import config_hash, load_config
from elsc.data.cache_dataset import AuxiliaryCache, lexical_tensors_from_batch
from elsc.data.cico_dataset import CiCoFeatureDataset
from elsc.data.tokenize import TEXT_AUGMENTATION_RECIPE, CiCoCollator
from elsc.losses.coarse import balanced_clcl_loss
from elsc.losses.lexical import globally_normalized_auxiliary, lexical_loss
from elsc.mining.build_cache import mining_config_hash
from elsc.provenance import validate_dev_selection
from elsc.resources import require_resources
from elsc.train import (
    _amp_settings,
    _evidence_objective,
    _in_batch_word_negative_mask,
    _keep_objective,
    _optimizer,
    _seed_everything,
)
from elsc.upstream.factory import build_retriever_from_checkpoint, load_cico_tokenizer
from elsc.utils import atomic_json_dump, sha256_file


def _stress_batch_indices(
    pair_ids: list[str], eligible_pair_ids: set[str], batch_size: int
) -> list[int]:
    eligible = [index for index, pair_id in enumerate(pair_ids) if pair_id in eligible_pair_ids]
    other = [index for index, pair_id in enumerate(pair_ids) if pair_id not in eligible_pair_ids]
    indexes = (eligible + other)[:batch_size]
    if len(indexes) != batch_size:
        raise ValueError("training manifest is smaller than the preflight batch")
    return indexes


def _load_auxiliary_cache(
    config: dict[str, Any], device: torch.device
) -> tuple[AuxiliaryCache, torch.Tensor]:
    teacher_path = Path(config["model"]["teacher_checkpoint"])
    validate_dev_selection(config["model"]["teacher_selection_provenance"], teacher_path)
    if config["model"].get("require_init_equals_teacher", False):
        initialization = Path(config["model"]["init_checkpoint"])
        if sha256_file(initialization) != sha256_file(teacher_path):
            raise ValueError("preflight requires student initialization = teacher")
    cache = AuxiliaryCache(
        config["cache"]["path"],
        {
            "split": "train",
            "teacher_hash": sha256_file(teacher_path),
            "manifest_hash": sha256_file(config["data"]["train_manifest"]),
            "language": config["data"]["caption_language"],
            "feature_fusion": f"sum:{config['data']['alpha']}",
            "tokenizer_hash": sha256_file(
                Path(config["upstream"]["cico_root"]) / "modules" / "bpe_simple_vocab_16e6.txt.gz"
            ),
            "view_sampling": "upstream_uniform+jitter1_v1",
            "mining_config_hash": mining_config_hash(config),
            "negative_source": config["cache"].get("negatives_source", "train_visual_neighbors_v1"),
        },
    )
    lexical_bank = (
        torch.from_numpy(np.load(Path(config["cache"]["path"]) / "lexical_bank.npy"))
        .float()
        .to(device)
    )
    return cache, lexical_bank


def run_preflight(
    config: dict[str, Any],
    *,
    device: torch.device,
    min_free_after_gib: float,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("training preflight requires an available CUDA device")
    if config.get("method") not in {"baseline", "elsc", "local_word_video"}:
        raise ValueError("training preflight does not yet support this method objective")
    if float(config.get("caption", {}).get("weight", 0.0)) > 0:
        raise ValueError("training preflight must include caption loss before using its config")
    resources = config.get("resources", {})
    initial = require_resources(
        Path(config["data"]["train_manifest"]).parent,
        device,
        min_disk_gib=float(resources.get("min_free_disk_gib", 20)),
        min_gpu_gib=float(resources.get("min_free_gpu_gib_train", 12)),
        operation="training batch preflight",
    )
    _seed_everything(int(config["seed"]))
    checkpoint = Path(config["model"]["init_checkpoint"])
    model, _ = build_retriever_from_checkpoint(config, checkpoint, device=device)
    tokenizer = load_cico_tokenizer(config)
    dataset = CiCoFeatureDataset(
        config["data"]["train_manifest"],
        feature_len=int(config["data"]["feature_len"]),
        alpha=float(config["data"]["alpha"]),
        split="train",
    )
    batch_size = int(config["train"]["per_device_batch"])
    cache = None
    lexical_bank = None
    teacher = None
    eligible_pair_ids: set[str] = set()
    if config.get("method") != "baseline":
        cache, lexical_bank = _load_auxiliary_cache(config, device)
        eligible_pair_ids = {
            pair_id
            for pair_id, records in cache.by_pair.items()
            if any(record.get("evidence_eligible", False) for record in records)
        }
        if config.get("evidence", {}).get("enabled") and not eligible_pair_ids:
            raise ValueError("ELSC-Full preflight cache has no evidence-eligible records")
        if float(config.get("keep", {}).get("weight", 0.0)) > 0:
            teacher_path = Path(config["model"]["teacher_checkpoint"])
            teacher, _ = build_retriever_from_checkpoint(config, teacher_path, device=device)
            teacher.eval().requires_grad_(False)
    pair_ids = [str(record.pair_id) for record in dataset.records]
    indexes = _stress_batch_indices(pair_ids, eligible_pair_ids, batch_size)
    collator = CiCoCollator(
        tokenizer,
        int(config["data"]["max_words"]),
        augment=(config["data"].get("text_augmentation") == TEXT_AUGMENTATION_RECIPE),
        seed=int(config["seed"]),
    )
    batch = collator([dataset[index] for index in indexes])
    optimizer = _optimizer(model, config)
    amp_enabled, amp_dtype = _amp_settings(config, device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    h = batch["h"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    dense_index = batch["dense_index"].to(device, non_blocking=True)
    clean_inputs = tuple(value.to(device, non_blocking=True) for value in batch["clean_text"])
    aug_inputs = tuple(value.to(device, non_blocking=True) for value in batch["aug_text"])
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        video, h_prime = model.encode_video(h, valid)
        clean_text = model.encode_text(*clean_inputs)
        aug_text = model.encode_text(*aug_inputs)
        i2t, t2i = model.bridge.score(video, clean_text, text_aug=aug_text, objective=True)
        coarse = balanced_clcl_loss(
            i2t,
            t2i,
            dual_mix=float(config["model"]["dual_mix"]),
            mix_design=config["model"]["mix_design"],
        )
        lexical = h_prime.sum() * 0.0
        lexical_count = torch.zeros((), dtype=torch.long, device=device)
        dependence = h_prime.sum() * 0.0
        invariance = h_prime.sum() * 0.0
        keep = h_prime.sum() * 0.0
        evidence_count = 0
        evidence_diagnostics: dict[str, float | int | None] = {
            "rho_mean": None,
            "clean_margin_mean": None,
            "evidence_removed_margin_mean": None,
            "control_removed_margin_mean": None,
            "video_encoder_calls": 0,
            "paired_score_calls": 0,
            "negative_text_encoder_calls": 0,
        }
        auxiliary_records = 0
        if cache is not None and lexical_bank is not None:
            records_by_sample = cache.lookup(batch["pair_id"], batch["view_hash"])
            auxiliary_records = sum(len(records) for records in records_by_sample)
            if config.get("method") in {"elsc", "local_word_video"}:
                z = model.local_head(h_prime)
                tensors = lexical_tensors_from_batch(
                    z,
                    dense_index,
                    records_by_sample,
                    lexical_bank,
                    support_mode=config.get("aux_support_mode", "teacher"),
                    seed=int(config["seed"]),
                    random_span_duration_tolerance=float(
                        config["cache"].get("random_span_duration_tolerance", 0.10)
                    ),
                )
                (
                    support_z,
                    support_weights,
                    positive_e,
                    negative_e,
                    negative_valid,
                    rho,
                    word_ids,
                ) = tensors
                if config.get("method") == "local_word_video" and len(positive_e) > 1:
                    occurrence_count = len(positive_e)
                    negative_e = positive_e.detach()[None].expand(occurrence_count, -1, -1)
                    negative_valid = _in_batch_word_negative_mask(word_ids)
                elif config.get("method") == "local_word_video":
                    negative_valid = torch.zeros_like(negative_valid)
                lexical_sum, lexical_count = lexical_loss(
                    support_z,
                    support_weights,
                    positive_e,
                    negative_e,
                    negative_valid,
                    rho,
                    margin=float(config["loss"]["lexical_margin"]),
                    temperature=float(config["loss"]["lexical_temperature"]),
                )
                lexical = globally_normalized_auxiliary(lexical_sum, lexical_count)
            if config.get("evidence", {}).get("enabled"):
                dependence, invariance, evidence_count, evidence_diagnostics = _evidence_objective(
                    model,
                    tokenizer,
                    h,
                    valid,
                    dense_index,
                    video,
                    clean_text,
                    records_by_sample,
                    config,
                    device,
                    epoch=0,
                    step=0,
                )
                expected_evidence = max(
                    1,
                    int(batch_size * float(config["evidence"]["batch_fraction"])),
                )
                eligible_candidates = sum(
                    record.get("evidence_eligible", False)
                    for records in records_by_sample
                    for record in records
                )
                exercised_evidence = min(expected_evidence, eligible_candidates)
                if evidence_count != exercised_evidence:
                    raise ValueError(
                        "preflight stress batch did not exercise the expected evidence count: "
                        f"{evidence_count} != {exercised_evidence}"
                    )
            if teacher is not None:
                keep = _keep_objective(
                    model,
                    teacher,
                    h,
                    valid,
                    clean_inputs,
                    video,
                    clean_text,
                    dual_mix=float(config["model"]["dual_mix"]),
                    temperature=float(config["keep"].get("temperature", 1.0)),
                )
        loss = (
            coarse
            + float(config["loss"]["lexical_weight"]) * lexical
            + float(config.get("evidence", {}).get("dependence_weight", 0.0)) * dependence
            + float(config.get("evidence", {}).get("invariance_weight", 0.0)) * invariance
            + float(config.get("keep", {}).get("weight", 0.0)) * keep
        )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("training preflight objective is non-finite")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(config["train"]["grad_clip_norm"]),
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    free_after, total = torch.cuda.mem_get_info(device)
    required_after = float(min_free_after_gib) * 1024**3
    status = "passed" if free_after >= required_after else "failed_gpu_headroom"
    return {
        "schema_version": 1,
        "status": status,
        "config_hash": config_hash(config),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(Path(config["data"]["train_manifest"]).resolve()),
        "manifest_sha256": sha256_file(config["data"]["train_manifest"]),
        "batch_size": batch_size,
        "negative_pool_per_query": batch_size - 1,
        "precision": config["train"].get("precision", "fp32"),
        "loss": float(loss.detach()),
        "loss_components": {
            "coarse": float(coarse.detach()),
            "lexical": float(lexical.detach()),
            "dependence": float(dependence.detach()),
            "invariance": float(invariance.detach()),
            "keep": float(keep.detach()),
        },
        "lexical_count": int(lexical_count),
        "evidence_count": evidence_count,
        "teacher_loaded_for_keep": teacher is not None,
        "auxiliary_record_count": auxiliary_records,
        "evidence_diagnostics": evidence_diagnostics,
        "stress_batch_eligible_pair_count": sum(
            pair_id in eligible_pair_ids for pair_id in batch["pair_id"]
        ),
        "evidence_activation_checkpoint": bool(
            config.get("evidence", {}).get("activation_checkpoint", False)
        ),
        "evidence_encoder_microbatch_size": config.get("evidence", {}).get(
            "encoder_microbatch_size"
        ),
        "evidence_score_microbatch_size": config.get("evidence", {}).get("score_microbatch_size"),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_free_after_bytes": int(free_after),
        "gpu_total_bytes": int(total),
        "required_free_after_bytes": int(required_after),
        "initial_resources": initial,
        "optimizer_state_initialized": bool(optimizer.state),
        "data_scope": "train_only",
        "batch_selection": (
            "train_only_evidence_eligible_stress_batch"
            if eligible_pair_ids
            else "train_only_first_deterministic_batch"
        ),
        "checkpoint_written": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure one real train-only CiCo optimizer step before a long run"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--min-free-after-gib", type=float, default=4.0)
    args = parser.parse_args(argv)
    if args.min_free_after_gib < 0:
        raise ValueError("min-free-after-gib must be nonnegative")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output = Path(args.output)
    try:
        config = load_config(args.config, stage="train")
        if args.batch_size is not None:
            config["train"]["per_device_batch"] = args.batch_size
        result = run_preflight(
            config,
            device=torch.device(args.device),
            min_free_after_gib=args.min_free_after_gib,
        )
    except torch.cuda.OutOfMemoryError as error:
        device = torch.device(args.device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        torch.cuda.empty_cache()
        free_after_cleanup, total = torch.cuda.mem_get_info(device)
        result = {
            "schema_version": 1,
            "status": "failed_cuda_oom",
            "config": str(Path(args.config).resolve()),
            "config_hash": config_hash(config),
            "batch_size": int(config["train"]["per_device_batch"]),
            "precision": config["train"].get("precision", "fp32"),
            "error": repr(error),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "gpu_free_after_cleanup_bytes": int(free_after_cleanup),
            "gpu_total_bytes": int(total),
            "checkpoint_written": False,
            "data_scope": "train_only_first_deterministic_batch",
        }
    except FloatingPointError as error:
        result = {
            "schema_version": 1,
            "status": "failed_nonfinite_objective",
            "config": str(Path(args.config).resolve()),
            "config_hash": config_hash(config),
            "batch_size": int(config["train"]["per_device_batch"]),
            "precision": config["train"].get("precision", "fp32"),
            "error": repr(error),
            "checkpoint_written": False,
            "data_scope": "train_only",
        }
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
