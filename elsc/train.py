from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from elsc.config import config_hash, dump_resolved, load_config
from elsc.data.cache_dataset import AuxiliaryCache, lexical_tensors_from_batch
from elsc.data.cico_dataset import CiCoFeatureDataset
from elsc.data.tokenize import TEXT_AUGMENTATION_RECIPE, CiCoCollator, encode_cico_text
from elsc.evaluate import evaluate_model
from elsc.losses.caption import matched_caption_loss
from elsc.losses.coarse import balanced_clcl_loss
from elsc.losses.evidence import apply_input_intervention, evidence_losses
from elsc.losses.distillation import bidirectional_kl
from elsc.losses.lexical import globally_normalized_auxiliary, lexical_loss
from elsc.mining.build_cache import mining_config_hash
from elsc.provenance import validate_dev_selection
from elsc.resources import require_resources
from elsc.upstream.cico_bridge import TextEncoding, VideoEncoding
from elsc.upstream.factory import build_retriever_from_checkpoint, load_cico_tokenizer
from elsc.utils import (
    atomic_json_dump,
    capture_rng_state,
    restore_rng_state,
    sha256_file,
    stable_seed,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _amp_settings(config: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    precision = config["train"].get("precision", "fp32")
    if precision == "fp32":
        return False, torch.float32
    if device.type != "cuda":
        raise ValueError(f"{precision} requires a CUDA device")
    if precision == "amp_bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("amp_bf16 requested but this GPU does not support bfloat16")
        return True, torch.bfloat16
    return True, torch.float16


def _slice_video(value: VideoEncoding, index: int) -> VideoEncoding:
    return VideoEncoding(
        value.mask[index : index + 1], value.tokens[index : index + 1], value.cls[index : index + 1]
    )


def _slice_text(value: TextEncoding, index: int) -> TextEncoding:
    return TextEncoding(
        value.mask[index : index + 1], value.tokens[index : index + 1], value.cls[index : index + 1]
    )


def _caption_objective(
    model,
    tokenizer,
    video: VideoEncoding,
    clean_text: TextEncoding,
    records_by_sample: list[list[dict[str, Any]]],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives: list[torch.Tensor] = []
    negative_rows: list[torch.Tensor] = []
    reliabilities: list[float] = []
    max_negatives = max(
        (
            len(record.get("negative_captions", []))
            for records in records_by_sample
            for record in records
        ),
        default=0,
    )
    if max_negatives == 0:
        zero = video.tokens.sum() * 0.0
        return zero, torch.zeros((), dtype=torch.long, device=device)
    scale = model.core.clip.logit_scale.exp().detach().float()
    dual_mix = float(config["model"]["dual_mix"])
    for sample, records in enumerate(records_by_sample):
        for record in records:
            captions = record.get("negative_captions", [])
            if not captions:
                continue
            own_video = _slice_video(video, sample)
            own_text = _slice_text(clean_text, sample)
            pos_i2t, pos_t2i = model.bridge.score(own_video, own_text, objective=True)
            positives.append(
                model.bridge.mixed_score(pos_i2t, pos_t2i, dual_mix).reshape(()) / scale
            )
            encoded = [
                encode_cico_text(text, tokenizer, int(config["data"]["max_words"]))
                for text in captions
            ]
            ids, segments, masks = (
                torch.stack([item[index] for item in encoded]).to(device) for index in range(3)
            )
            negative_text = model.encode_text(ids, segments, masks)
            neg_i2t, neg_t2i = model.bridge.score(own_video, negative_text, objective=True)
            scores = model.bridge.mixed_score(neg_i2t, neg_t2i, dual_mix)[0] / scale
            row = scores.new_full((max_negatives,), float("-inf"))
            row[: len(scores)] = scores
            negative_rows.append(row)
            reliabilities.append(float(record["rho"]))
    negative_scores = torch.stack(negative_rows)
    valid = torch.isfinite(negative_scores)
    return matched_caption_loss(
        torch.stack(positives),
        negative_scores,
        valid,
        torch.tensor(reliabilities, device=device),
        margin=float(config["caption"]["margin"]),
        temperature=float(config["caption"]["temperature"]),
    )


def _paired_margin(
    model, video: VideoEncoding, positive: TextEncoding, negative: TextEncoding, dual_mix: float
) -> torch.Tensor:
    text = TextEncoding(
        torch.cat((positive.mask, negative.mask)),
        torch.cat((positive.tokens, negative.tokens)),
        torch.cat((positive.cls, negative.cls)),
    )
    i2t, t2i = model.bridge.score(video, text, objective=True)
    scores = model.bridge.mixed_score(i2t, t2i, dual_mix)[0]
    return (scores[0] - scores[1]) / model.core.clip.logit_scale.exp().detach().float()


def _evidence_objective(
    model,
    tokenizer,
    h: torch.Tensor,
    valid: torch.Tensor,
    dense_index: torch.Tensor,
    clean_video: VideoEncoding,
    clean_text: TextEncoding,
    records_by_sample: list[list[dict[str, Any]]],
    config: dict[str, Any],
    device: torch.device,
    *,
    epoch: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float | None]]:
    candidates = [
        (sample, record)
        for sample, records in enumerate(records_by_sample)
        for record in records
        if record.get("evidence_eligible", False)
    ]
    if not candidates:
        zero = h.sum() * 0.0
        return (
            zero,
            zero,
            0,
            {
                "rho_mean": None,
                "clean_margin_mean": None,
                "evidence_removed_margin_mean": None,
                "control_removed_margin_mean": None,
            },
        )
    candidates.sort(
        key=lambda item: stable_seed(
            config["seed"], epoch, step, item[1]["pair_id"], item[1]["word_id"]
        )
    )
    maximum = max(1, int(len(valid) * float(config["evidence"]["batch_fraction"])))
    selected = candidates[:maximum]
    clean_margins: list[torch.Tensor] = []
    evidence_margins: list[torch.Tensor] = []
    control_margins: list[torch.Tensor] = []
    reliability: list[float] = []
    for sample, record in selected:
        position_by_dense = {
            int(value): position
            for position, value in enumerate(dense_index[sample].tolist())
            if value >= 0
        }
        evidence_mask = torch.zeros_like(valid[sample])
        control_mask = torch.zeros_like(valid[sample])
        for value in record["evidence_remove_dense_indices"]:
            evidence_mask[position_by_dense[int(value)]] = True
        for value in record["control_remove_dense_indices"]:
            control_mask[position_by_dense[int(value)]] = True
        if int(evidence_mask.sum()) != int(control_mask.sum()):
            raise ValueError(f"cache W/C token count mismatch for {record['pair_id']}")
        own_h = h[sample : sample + 1]
        own_valid = valid[sample : sample + 1]
        fill = 0.0
        h_evidence = apply_input_intervention(own_h, evidence_mask[None], own_valid, fill)
        h_control = apply_input_intervention(own_h, control_mask[None], own_valid, fill)
        video_evidence, _ = model.encode_video(h_evidence, own_valid)
        video_control, _ = model.encode_video(h_control, own_valid)
        positive = _slice_text(clean_text, sample)
        encoded = encode_cico_text(
            record["negative_captions"][0], tokenizer, int(config["data"]["max_words"])
        )
        negative = model.encode_text(*(value.unsqueeze(0).to(device) for value in encoded))
        clean_margins.append(
            _paired_margin(
                model,
                _slice_video(clean_video, sample),
                positive,
                negative,
                float(config["model"]["dual_mix"]),
            )
        )
        evidence_margins.append(
            _paired_margin(
                model, video_evidence, positive, negative, float(config["model"]["dual_mix"])
            )
        )
        control_margins.append(
            _paired_margin(
                model, video_control, positive, negative, float(config["model"]["dual_mix"])
            )
        )
        reliability.append(float(record["rho"]))
    clean_tensor = torch.stack(clean_margins)
    evidence_tensor = torch.stack(evidence_margins)
    control_tensor = torch.stack(control_margins)
    reliability_tensor = torch.tensor(reliability, device=device)
    dep, inv = evidence_losses(
        clean_tensor,
        evidence_tensor,
        control_tensor,
        reliability_tensor,
        dependence_margin=float(config["evidence"]["dependence_margin"]),
        huber_delta=float(config["evidence"]["huber_delta"]),
    )
    diagnostics = {
        "rho_mean": float(reliability_tensor.mean()),
        "clean_margin_mean": float(clean_tensor.detach().float().mean()),
        "evidence_removed_margin_mean": float(evidence_tensor.detach().float().mean()),
        "control_removed_margin_mean": float(control_tensor.detach().float().mean()),
    }
    return dep, inv, len(selected), diagnostics


def _optimizer(model, config: dict[str, Any]):
    train_cfg = config["train"]
    groups: list[dict[str, Any]] = []
    decay = float(train_cfg["weight_decay"])

    def add_groups(name: str, named_parameters, lr: float) -> None:
        materialized = [
            (parameter_name, parameter)
            for parameter_name, parameter in named_parameters
            if parameter.requires_grad
        ]
        if train_cfg.get("exclude_bias_and_1d_from_weight_decay", True):
            decayed = [
                parameter
                for parameter_name, parameter in materialized
                if parameter.ndim > 1 and not parameter_name.endswith(".bias")
            ]
            no_decay = [
                parameter
                for parameter_name, parameter in materialized
                if parameter.ndim <= 1 or parameter_name.endswith(".bias")
            ]
        else:
            decayed = [parameter for _, parameter in materialized]
            no_decay = []
        if decayed:
            groups.append(
                {"params": decayed, "lr": lr, "weight_decay": decay, "name": f"{name}_decay"}
            )
        if no_decay:
            groups.append(
                {"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{name}_no_decay"}
            )

    if config["model"].get("logit_scale_frozen", False):
        model.core.clip.logit_scale.requires_grad_(False)
    if not config["model"].get("backbone_frozen", True):
        add_groups("core", model.core.named_parameters(), float(train_cfg["core_lr"]))
    adapter_trainable = config["model"]["adapter"].get("trainable", True)
    model.adapter.requires_grad_(bool(adapter_trainable and model.adapter_enabled))
    adapter_parameters = [
        parameter for parameter in model.adapter.parameters() if parameter.requires_grad
    ]
    if adapter_parameters:
        add_groups("adapter", model.adapter.named_parameters(), float(train_cfg["adapter_lr"]))
    head_required = config.get("method") in {"elsc", "local_word_video"}
    model.local_head.requires_grad_(head_required)
    if head_required:
        add_groups(
            "local_head", model.local_head.named_parameters(), float(train_cfg["head_lr"])
        )
    if not groups:
        raise ValueError("all model parameters are frozen")
    return torch.optim.AdamW(
        groups,
        lr=float(train_cfg["core_lr"]),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.98))),
        eps=float(train_cfg.get("epsilon", 1e-6)),
        weight_decay=decay,
    )


def _scheduler(optimizer, steps: int, warmup_ratio: float):
    warmup = int(steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    sampler_generator: torch.Generator,
    epoch: int,
    step: int,
    config: dict,
    best: float,
    provenance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "elsc-training-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_dev": best,
            "rng": capture_rng_state(),
            "sampler_generator_state": sampler_generator.get_state(),
            "config_hash": config_hash(config),
            "trainable_parameter_names": [
                name for name, value in model.named_parameters() if value.requires_grad
            ],
            "provenance": provenance,
        },
        temporary,
    )
    temporary.replace(path)


def train(
    config: dict[str, Any], run_dir: Path, *, device: torch.device, resume: Path | None = None
) -> dict[str, Any]:
    resources = config.get("resources", {})
    initial_resources = require_resources(
        run_dir.parent,
        device,
        min_disk_gib=float(resources.get("min_free_disk_gib", 20)),
        min_gpu_gib=float(resources.get("min_free_gpu_gib_train", 12)),
        operation="training",
    )
    _seed_everything(int(config["seed"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    init_checkpoint = resume or Path(config["model"]["init_checkpoint"])
    model, raw_checkpoint = build_retriever_from_checkpoint(config, init_checkpoint, device=device)
    tokenizer = load_cico_tokenizer(config)
    dataset = CiCoFeatureDataset(
        config["data"]["train_manifest"],
        feature_len=int(config["data"]["feature_len"]),
        alpha=float(config["data"]["alpha"]),
        split="train",
    )
    batch_size = int(config["train"]["per_device_batch"])
    if len(dataset) < batch_size:
        raise ValueError(
            f"training manifest has {len(dataset)} records, fewer than the configured "
            f"drop-last batch size {batch_size}"
        )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    collator = CiCoCollator(
        tokenizer,
        int(config["data"]["max_words"]),
        augment=config["data"].get("text_augmentation") == TEXT_AUGMENTATION_RECIPE,
        seed=int(config["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    if len(loader) == 0:
        raise ValueError("training DataLoader has zero batches")
    optimizer = _optimizer(model, config)
    amp_enabled, amp_dtype = _amp_settings(config, device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    accumulation = int(config["train"].get("accumulation_steps", 1))
    optimizer_steps = math.ceil(len(loader) / accumulation) * int(config["train"]["epochs"])
    scheduler = _scheduler(optimizer, optimizer_steps, float(config["train"]["warmup_ratio"]))
    start_epoch = 0
    global_step = 0
    best_dev = -math.inf
    best_dev_r5 = -math.inf
    if resume is not None and raw_checkpoint.get("format") == "elsc-training-v1":
        if raw_checkpoint.get("config_hash") != config_hash(config):
            raise ValueError("resume checkpoint config hash differs from the resolved run config")
        optimizer.load_state_dict(raw_checkpoint["optimizer"])
        scheduler.load_state_dict(raw_checkpoint["scheduler"])
        if scaler.is_enabled() and "amp_scaler" not in raw_checkpoint:
            raise ValueError("fp16 resume checkpoint is missing AMP scaler state")
        if "amp_scaler" in raw_checkpoint:
            scaler.load_state_dict(raw_checkpoint["amp_scaler"])
        start_epoch = int(raw_checkpoint["epoch"]) + 1
        global_step = int(raw_checkpoint["step"])
        best_dev = float(raw_checkpoint["best_dev"])
        metrics_path = run_dir / "best_dev_metrics.json"
        if metrics_path.is_file():
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
            best_dev_r5 = 0.5 * (float(previous["V2T"]["R5"]) + float(previous["T2V"]["R5"]))
        restore_rng_state(raw_checkpoint["rng"])
        generator.set_state(raw_checkpoint["sampler_generator_state"])

    dump_resolved(config, run_dir / "resolved_config.yaml")

    provenance: dict[str, Any] = {
        "upstream_commit": config["upstream"]["cico_commit"],
        "initialization": {
            "path": str(config["model"]["init_checkpoint"]),
            "sha256": sha256_file(config["model"]["init_checkpoint"]),
        },
        "manifests": {
            split: sha256_file(config["data"][f"{split}_manifest"]) for split in ("train", "dev")
        },
        "selection_rule": {
            "split": "dev",
            "primary": config["train"]["checkpoint_metric"],
            "tie_break": "mean_t2v_v2t_r5_then_earlier_epoch",
            "test_access_during_training": False,
        },
        "text_augmentation": {
            "recipe": config["data"].get("text_augmentation"),
            "decision_seed": "sha256(seed,epoch,pair_id,recipe)",
            "historical_rng_trace_claimed": False,
        },
        "initial_resources": initial_resources,
    }
    teacher_value = config["model"].get("teacher_checkpoint")
    if teacher_value:
        provenance["teacher"] = {"path": str(teacher_value), "sha256": sha256_file(teacher_value)}
    if config.get("method") != "baseline":
        cache_meta_path = Path(config["cache"]["path"]) / "cache_meta.json"
        provenance["cache_meta_sha256"] = sha256_file(cache_meta_path)
    atomic_json_dump(provenance, run_dir / "provenance.json")

    cache = None
    lexical_bank = None
    teacher = None
    if config.get("method") != "baseline":
        teacher_path = Path(config["model"]["teacher_checkpoint"])
        validate_dev_selection(config["model"]["teacher_selection_provenance"], teacher_path)
        if config["model"].get("require_init_equals_teacher", False):
            configured_initialization = Path(config["model"]["init_checkpoint"])
            if sha256_file(configured_initialization) != sha256_file(teacher_path):
                raise ValueError("this ELSC-Min run requires student initialization = teacher")
        cache = AuxiliaryCache(
            config["cache"]["path"],
            {
                "split": "train",
                "teacher_hash": sha256_file(teacher_path),
                "manifest_hash": sha256_file(config["data"]["train_manifest"]),
                "language": config["data"]["caption_language"],
                "feature_fusion": f"sum:{config['data']['alpha']}",
                "tokenizer_hash": sha256_file(
                    Path(config["upstream"]["cico_root"])
                    / "modules"
                    / "bpe_simple_vocab_16e6.txt.gz"
                ),
                "view_sampling": "upstream_uniform+jitter1_v1",
                "mining_config_hash": mining_config_hash(config),
                "negative_source": config["cache"].get(
                    "negatives_source", "train_visual_neighbors_v1"
                ),
            },
        )
        lexical_bank = (
            torch.from_numpy(np.load(Path(config["cache"]["path"]) / "lexical_bank.npy"))
            .float()
            .to(device)
        )
        if config.get("evidence", {}).get("enabled") and not any(
            record.get("evidence_eligible", False)
            for records in cache.by_pair.values()
            for record in records
        ):
            raise ValueError("ELSC-Full requested but cache contains no evidence-eligible records")
        if float(config.get("keep", {}).get("weight", 0.0)) > 0:
            teacher, _ = build_retriever_from_checkpoint(config, teacher_path, device=device)
            teacher.eval().requires_grad_(False)

    log_path = run_dir / "train.jsonl"
    started = time.time()
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        collator.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            h = batch["h"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            dense_index = batch["dense_index"].to(device, non_blocking=True)
            clean_inputs = tuple(
                value.to(device, non_blocking=True) for value in batch["clean_text"]
            )
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
            zero = h_prime.sum() * 0.0
            lexical = zero
            lexical_count = torch.zeros((), dtype=torch.long, device=device)
            caption = zero
            caption_count = torch.zeros((), dtype=torch.long, device=device)
            dependence = zero
            invariance = zero
            keep = zero
            evidence_count = 0
            evidence_diagnostics: dict[str, float | None] = {
                "rho_mean": None,
                "clean_margin_mean": None,
                "evidence_removed_margin_mean": None,
                "control_removed_margin_mean": None,
            }
            batch_aux_records = 0
            batch_aux_rho_mean = None
            lexical_gradient_diagnostic = None
            if cache is not None:
                records_by_sample = cache.lookup(batch["pair_id"], batch["view_hash"])
                batch_records = [record for records in records_by_sample for record in records]
                batch_aux_records = len(batch_records)
                if batch_records:
                    batch_aux_rho_mean = float(
                        np.mean([float(record["rho"]) for record in batch_records])
                    )
                if config.get("method") in {"elsc", "local_word_video"}:
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        z = model.local_head(h_prime)
                    tensors = lexical_tensors_from_batch(
                        z,
                        dense_index,
                        records_by_sample,
                        lexical_bank,
                        support_mode=config.get("aux_support_mode", "teacher"),
                        seed=int(config["seed"]),
                    )
                    support_z, support_weights, positive_e, negative_e, negative_valid, rho = (
                        tensors
                    )
                    if config.get("method") == "local_word_video" and len(positive_e) > 1:
                        occurrence_count = len(positive_e)
                        negative_e = positive_e.detach()[None].expand(occurrence_count, -1, -1)
                        negative_valid = ~torch.eye(
                            occurrence_count, dtype=torch.bool, device=device
                        )
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
                    diagnostic_interval = int(
                        config["train"].get("gradient_diagnostic_interval", 100)
                    )
                    if int(lexical_count) > 0 and batch_index % diagnostic_interval == 0:
                        diagnostic_parameters = (
                            model.adapter.up.weight,
                            model.adapter.down.weight,
                            model.local_head.proj.weight,
                        )
                        diagnostic_gradients = torch.autograd.grad(
                            lexical,
                            diagnostic_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        lexical_gradient_diagnostic = {
                            name: (
                                float(gradient.detach().float().norm())
                                if gradient is not None
                                else 0.0
                            )
                            for name, gradient in zip(
                                ("adapter_up", "adapter_down", "local_head"),
                                diagnostic_gradients,
                                strict=True,
                            )
                        }
                elif config.get("method") == "matched_caption":
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        caption_sum, caption_count = _caption_objective(
                            model, tokenizer, video, clean_text, records_by_sample, config, device
                        )
                    caption = globally_normalized_auxiliary(caption_sum, caption_count)
                if config.get("evidence", {}).get("enabled"):
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        dependence, invariance, evidence_count, evidence_diagnostics = (
                            _evidence_objective(
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
                                epoch=epoch,
                                step=global_step,
                            )
                        )
                if teacher is not None:
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                        ),
                    ):
                        teacher_video, _ = teacher.encode_video(h, valid)
                        teacher_text = teacher.encode_text(*clean_inputs)
                        teacher_i2t, teacher_t2i = teacher.bridge.score(
                            teacher_video, teacher_text, objective=True
                        )
                        teacher_scores = teacher.bridge.mixed_score(
                            teacher_i2t, teacher_t2i, float(config["model"]["dual_mix"])
                        )
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        student_clean_i2t, student_clean_t2i = model.bridge.score(
                            video, clean_text, objective=True
                        )
                        student_scores = model.bridge.mixed_score(
                            student_clean_i2t,
                            student_clean_t2i,
                            float(config["model"]["dual_mix"]),
                        )
                    keep = bidirectional_kl(
                        teacher_scores,
                        student_scores,
                        temperature=float(config["keep"].get("temperature", 1.0)),
                    )
            ramp = min(
                1.0,
                (epoch + (batch_index + 1) / len(loader))
                / max(1, int(config["loss"].get("lexical_weight_ramp_epochs", 1))),
            )
            loss = (
                coarse
                + ramp * float(config["loss"]["lexical_weight"]) * lexical
                + float(config["caption"].get("weight", 0.0)) * caption
                + float(config.get("evidence", {}).get("dependence_weight", 0.0)) * dependence
                + float(config.get("evidence", {}).get("invariance_weight", 0.0)) * invariance
                + float(config.get("keep", {}).get("weight", 0.0)) * keep
            )
            window_start = (batch_index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            scaler.scale(loss / window_size).backward()
            update_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            gradient_norms = None
            if update_step:
                scaler.unscale_(optimizer)
                gradient_norms = {
                    "adapter_up": float(model.adapter.up.weight.grad.norm())
                    if model.adapter.up.weight.grad is not None
                    else 0.0,
                    "adapter_down": float(model.adapter.down.weight.grad.norm())
                    if model.adapter.down.weight.grad is not None
                    else 0.0,
                    "local_head": float(model.local_head.proj.weight.grad.norm())
                    if model.local_head.proj.weight.grad is not None
                    else 0.0,
                }
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(config["train"]["grad_clip_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            record = {
                "epoch": epoch,
                "batch": batch_index,
                "step": global_step,
                "loss": float(loss.detach()),
                "coarse": float(coarse.detach()),
                "lexical": float(lexical.detach()),
                "caption": float(caption.detach()),
                "dependence": float(dependence.detach()),
                "invariance": float(invariance.detach()),
                "keep": float(keep.detach()),
                "lexical_count": int(lexical_count),
                "caption_count": int(caption_count),
                "evidence_count": evidence_count,
                "aux_record_count": batch_aux_records,
                "aux_rho_mean": batch_aux_rho_mean,
                "evidence_diagnostics": evidence_diagnostics,
                "lr": [group["lr"] for group in optimizer.param_groups],
                "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
                "gradient_norms": gradient_norms,
                "lexical_gradient_diagnostic": lexical_gradient_diagnostic,
                "precision": config["train"].get("precision", "fp32"),
                "amp_scale": float(scaler.get_scale()),
                "adapter_residual_ratio": float(
                    ((h_prime - h).float().norm() / h.float().norm().clamp_min(1e-12)).detach()
                ),
                "actual_global_negative_pool_per_query": len(batch["pair_id"]) - 1,
                "effective_optimizer_batch": len(batch["pair_id"]) * window_size,
                "forward_counts": {
                    "student_video": 1 + 2 * evidence_count,
                    "student_text": 2 + int(caption_count) + evidence_count,
                    "teacher_video": 1 if teacher is not None else 0,
                    "teacher_text": 1 if teacher is not None else 0,
                },
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        _, metrics = evaluate_model(model, config, "dev", device)
        dev_score = 0.5 * (float(metrics["V2T"]["R1"]) + float(metrics["T2V"]["R1"]))
        dev_r5 = 0.5 * (float(metrics["V2T"]["R5"]) + float(metrics["T2V"]["R5"]))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "dev_evaluation",
                        "epoch": epoch,
                        "step": global_step,
                        "split": "dev",
                        "metrics": {direction: metrics[direction] for direction in ("V2T", "T2V")},
                        "elapsed_seconds": time.time() - started,
                        "peak_gpu_memory_bytes": (
                            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            generator,
            epoch,
            global_step,
            config,
            max(best_dev, dev_score),
            provenance,
        )
        if (dev_score, dev_r5) > (best_dev, best_dev_r5):
            best_dev = dev_score
            best_dev_r5 = dev_r5
            _save_checkpoint(
                run_dir / "checkpoints" / "best_dev.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                generator,
                epoch,
                global_step,
                config,
                best_dev,
                provenance,
            )
            atomic_json_dump(
                {
                    "epoch": epoch,
                    "metric": "mean_t2v_v2t_r1",
                    "value": best_dev,
                    "metrics": metrics,
                },
                run_dir / "best_dev_metrics.json",
            )
    selection = {
        "selection_split": "dev",
        "metric": config["train"]["checkpoint_metric"],
        "best_value": best_dev,
        "tie_break_mean_r5": best_dev_r5,
        "checkpoint": "checkpoints/best_dev.pt",
        "checkpoint_sha256": sha256_file(run_dir / "checkpoints" / "best_dev.pt"),
        "config_hash": config_hash(config),
        "dev_manifest_sha256": sha256_file(config["data"]["dev_manifest"]),
        "selected_epoch": json.loads(
            (run_dir / "best_dev_metrics.json").read_text(encoding="utf-8")
        )["epoch"],
        "test_used_for_selection": False,
    }
    atomic_json_dump(selection, run_dir / "selection.json")
    atomic_json_dump(
        {
            "schema_version": 1,
            "status": "complete",
            "optimizer_steps": global_step,
            "epochs_completed": int(config["train"]["epochs"]),
            "elapsed_seconds": time.time() - started,
            "precision": config["train"].get("precision", "fp32"),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "selection_sha256": sha256_file(run_dir / "selection.json"),
            "test_access_during_training": False,
        },
        run_dir / "run_summary.json",
    )
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train corrected CiCo baseline or ELSC with dev-only selection"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    config = load_config(args.config, stage="train")
    result = train(
        config,
        Path(args.run_dir),
        device=torch.device(args.device),
        resume=Path(args.resume) if args.resume else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
