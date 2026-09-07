from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from elsc.config import config_hash, load_config
from elsc.data.cico_dataset import CiCoFeatureDataset
from elsc.data.tokenize import TEXT_AUGMENTATION_RECIPE, CiCoCollator
from elsc.losses.coarse import balanced_clcl_loss
from elsc.resources import require_resources
from elsc.train import _amp_settings, _optimizer, _seed_everything
from elsc.upstream.factory import build_retriever_from_checkpoint, load_cico_tokenizer
from elsc.utils import atomic_json_dump, sha256_file


def run_preflight(
    config: dict[str, Any],
    *,
    device: torch.device,
    min_free_after_gib: float,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("training preflight requires an available CUDA device")
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=int(config["train"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=CiCoCollator(
            tokenizer,
            int(config["data"]["max_words"]),
            augment=(
                config["data"].get("text_augmentation") == TEXT_AUGMENTATION_RECIPE
            ),
            seed=int(config["seed"]),
        ),
    )
    if not len(loader):
        raise ValueError("training manifest is smaller than the preflight batch")
    optimizer = _optimizer(model, config)
    amp_enabled, amp_dtype = _amp_settings(config, device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp_enabled and amp_dtype == torch.float16
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(loader))
    h = batch["h"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    clean_inputs = tuple(value.to(device, non_blocking=True) for value in batch["clean_text"])
    aug_inputs = tuple(value.to(device, non_blocking=True) for value in batch["aug_text"])
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        video, _ = model.encode_video(h, valid)
        clean_text = model.encode_text(*clean_inputs)
        aug_text = model.encode_text(*aug_inputs)
        i2t, t2i = model.bridge.score(video, clean_text, text_aug=aug_text, objective=True)
        loss = balanced_clcl_loss(
            i2t,
            t2i,
            dual_mix=float(config["model"]["dual_mix"]),
            mix_design=config["model"]["mix_design"],
        )
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
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_free_after_bytes": int(free_after),
        "gpu_total_bytes": int(total),
        "required_free_after_bytes": int(required_after),
        "initial_resources": initial,
        "optimizer_state_initialized": bool(optimizer.state),
        "data_scope": "train_only_first_deterministic_batch",
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
        result = {
            "schema_version": 1,
            "status": "failed_cuda_oom",
            "config": str(Path(args.config).resolve()),
            "error": repr(error),
            "checkpoint_written": False,
            "data_scope": "train_only_first_deterministic_batch",
        }
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
