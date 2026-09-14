from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from pmgr.checkpoint import save_checkpoint, validate_resume
from pmgr.config import config_hash, load_config, resolved_with_overrides
from pmgr.data.group_dataset import GroupCollator, GroupDataset
from pmgr.data.group_sampler import GroupBatchSampler
from pmgr.distributed import coordinated_error, distributed_two_level_backward
from pmgr.evaluate import evaluate_model
from pmgr.losses import objective_loss
from pmgr.model import build_retriever, load_tokenizer
from pmgr.optimizer import build_inherited_bert_adam
from pmgr.replay import two_level_backward
from pmgr.scoring import mixed_pair_scores
from slr_common.utils import atomic_json_dump, git_worktree_state, restore_rng_state, sha256_file


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _load_group_items(dataset: GroupDataset, indexes: list[int], workers: int) -> list[dict[str, Any]]:
    """Load a group batch concurrently while preserving the sampler's dense order."""
    if workers < 1:
        raise ValueError("data workers must be positive")
    if workers == 1 or len(indexes) == 1:
        return [dataset[index] for index in indexes]
    with ThreadPoolExecutor(max_workers=min(workers, len(indexes))) as executor:
        return list(executor.map(dataset.__getitem__, indexes))


def _score(model, encoded, config):
    scoring = config["scoring"]
    return mixed_pair_scores(
        encoded.video_hidden,
        encoded.text_hidden,
        encoded.aug_hidden,
        encoded.video_valid,
        encoded.text_valid,
        encoded.aug_valid,
        omega=float(scoring["dual_mix"]),
        sigma=float(scoring["inner_temperature"]),
        normalize_eps=float(scoring["normalize_eps"]),
        mask_policy=scoring["mask_policy"],
    )


def _loss(model, q, a, b, batch, config):
    loss = config["loss"]
    return objective_loss(
        loss["mode"], q, batch["video_to_group"], model.logit_scale,
        batch["dataset_group_count"], batch["dataset_video_count"],
        a=a, b=b, omega=float(config["scoring"]["dual_mix"]),
        rank_mix=float(loss["rank_mix"]), rank_eta=float(loss["rank_eta"]),
    )


def _selection(metrics: dict[str, Any]) -> tuple[float, float]:
    primary = 0.5 * (float(metrics["T2V"]["R1"]) + float(metrics["V2T"]["R1"]))
    secondary = 0.25 * sum(
        float(metrics[direction][name])
        for direction in ("T2V", "V2T") for name in ("R5", "R10")
    )
    return primary, secondary


def train(
    config: dict[str, Any],
    output_dir: Path,
    *,
    device: torch.device,
    max_updates: int | None = None,
    resume: Path | None = None,
) -> dict[str, Any]:
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    if distributed and config["engine"]["distributed"] != "manual_sum_replay":
        raise ValueError("initialized workers require engine.distributed=manual_sum_replay")
    if distributed and int(config["training"]["effective_groups"]) % world_size:
        raise ValueError("effective group count must be divisible by worker count")
    _seed_everything(int(config["seed"]))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(config, output_dir / "resolved_config.json")
    if distributed:
        dist.barrier()
    checkpoint_path = resume or Path(config["paths"]["initialization_checkpoint"])
    model, raw = build_retriever(config, checkpoint=checkpoint_path, device=device)
    tokenizer = load_tokenizer(config)
    dataset = GroupDataset(
        config["paths"]["train_index"], feature_len=int(config["data"]["max_features"]),
        alpha=float(config["data"]["feature_mix_alpha"]),
    )
    mode = config["loss"]["mode"]
    representative = "uniform_deterministic" if mode in {"legacy_cico", "single_mixed_ce", "single_rank"} else "all"
    collator = GroupCollator(
        tokenizer, int(config["data"]["max_text_tokens"]),
        dataset_group_count=dataset.index.group_count,
        dataset_video_count=dataset.index.video_count,
        seed=int(config["seed"]),
        augment=bool(config["data"]["text_augmentation"]["enabled"]),
        augmentation_probability=float(config["data"]["text_augmentation"]["probability"]),
        representative_policy=representative,
    )
    sampler = GroupBatchSampler(
        len(dataset), int(config["training"]["effective_groups"]), seed=int(config["seed"]),
        drop_last=bool(config["data"]["drop_incomplete_group_batch"]),
    )
    total_steps = len(sampler) * int(config["training"]["epochs"])
    optimizer = build_inherited_bert_adam(model, config, total_steps)
    start_epoch = 0
    cursor = 0
    effective_step = 0
    best: dict[str, Any] | None = None
    if resume is not None:
        validate_resume(raw, config)
        optimizer.load_state_dict(raw["optimizer"])
        start_epoch = int(raw["epoch"])
        cursor = int(raw["sampler_cursor"])
        effective_step = int(raw["effective_step"])
        best = raw.get("best")
        restore_rng_state(raw["rng"])
    provenance = {
        "repository": git_worktree_state(Path.cwd()),
        "upstream_commit": config["upstream_commit"],
        "initialization": {
            "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
            "mode": "resume_exact" if resume else config["initialization_mode"],
        },
        "train_index_sha256": sha256_file(config["paths"]["train_index"]),
        "validation_index_sha256": sha256_file(config["paths"]["validation_index"]),
        "config_sha256": config_hash(config),
        "optimizer": "pinned_CiCo_BertAdam_warmup_cosine",
        "engine": config["engine"]["mode"],
        "distributed": config["engine"]["distributed"],
        "world_size": world_size,
        "data_workers_per_rank": int(config["training"]["num_workers"]),
    }
    log_path = output_dir / "train.jsonl"
    started = time.time()
    stopped = False
    last_batch: dict[str, Any] | None = None
    last_scores: torch.Tensor | None = None
    next_epoch, next_cursor = start_epoch, cursor
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        sampler.set_epoch(epoch, cursor=cursor if epoch == start_epoch else 0)
        for batch_position, indexes in enumerate(sampler, start=sampler.cursor):
            update_started = time.time()
            collator.set_position(epoch, effective_step)
            if distributed:
                groups_per_rank = len(indexes) // world_size
                local_indexes = indexes[rank * groups_per_rank : (rank + 1) * groups_per_rank]
            else:
                local_indexes = indexes
            load_started = time.time()
            load_error: BaseException | None = None
            batch: dict[str, Any] | None = None
            try:
                items = _load_group_items(
                    dataset, local_indexes, int(config["training"]["num_workers"])
                )
                batch = _device_batch(collator(items), device)
            except BaseException as error:
                load_error = error
            # All ranks reach the same failure collective before encoder/cache collectives.
            coordinated_error(load_error, device)
            assert batch is not None
            data_loading_seconds = time.time() - load_started
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if config["engine"]["mode"] == "direct":
                encoded = model.encode_pmgr_batch(batch)
                q, a, b = _score(model, encoded, config)
                output = _loss(model, q, a, b, batch, config)
                loss = output["loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("PMGR loss is nonfinite")
                loss.backward()
                encoder_forward_calls = 3
                score_block_replays = 0
            elif not distributed:
                replay = two_level_backward(model, batch, config)
                q = replay.q
                output = replay.diagnostics
                loss = replay.loss
                engine = config["engine"]
                encoder_chunks = (
                    math.ceil(len(batch["video_ids"]) / int(engine["video_encoder_microbatch"]))
                    + 2 * math.ceil(len(batch["group_ids"]) / int(engine["text_encoder_microbatch"]))
                )
                encoder_forward_calls = 2 * encoder_chunks
                score_block_replays = replay.block_replays
            else:
                replay = distributed_two_level_backward(model, batch, config)
                q = None
                output = replay.diagnostics
                loss = replay.loss
                encoder_forward_calls = replay.encoder_forward_calls
                score_block_replays = replay.block_replays
            active = [parameter for parameter in model.parameters() if parameter.grad is not None]
            if not active or any(not bool(torch.isfinite(parameter.grad).all()) for parameter in active):
                raise FloatingPointError("PMGR produced missing or nonfinite active gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(active, 1.0)
            optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(float(config["scoring"]["maximum_contrast_scale"])))
            effective_step += 1
            next_epoch, next_cursor = epoch, batch_position + 1
            if next_cursor == len(sampler):
                next_epoch, next_cursor = epoch + 1, 0
            record = {
                "effective_step": effective_step,
                "epoch": epoch,
                "batch": batch_position,
                "loss_mode": mode,
                "effective_groups": (
                    replay.global_group_count if distributed else len(batch["group_ids"])
                ),
                "loaded_videos": (
                    replay.global_video_count if distributed else len(batch["video_ids"])
                ),
                "candidate_pairs": (
                    replay.global_video_count * replay.global_group_count
                    if distributed else int(q.numel())
                ),
                "loss": float(loss.detach()),
                "ce_t": float(output["ce_t"].detach()),
                "ce_v": float(output["ce_v"].detach()),
                "rank_t": float(output["rank_t"].detach()),
                "rank_v": float(output["rank_v"].detach()),
                "contrast_scale": float(model.logit_scale.detach().exp()),
                "gradient_norm_before_clip": float(gradient_norm),
                "encoder_forward_calls": encoder_forward_calls,
                "score_block_replays": score_block_replays,
                "seconds_per_update": time.time() - update_started,
                "data_loading_seconds": data_loading_seconds,
                "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "group_ids": list(replay.group_ids) if distributed else batch["group_ids"],
                "video_ids": list(replay.video_ids) if distributed else batch["video_ids"],
                "full_group_sizes": (
                    list(replay.full_group_sizes)
                    if distributed else batch["full_group_sizes"].cpu().tolist()
                ),
                "selected_group_sizes": (
                    list(replay.selected_group_sizes)
                    if distributed else batch["selected_group_sizes"].cpu().tolist()
                ),
                "world_size": world_size,
            }
            if rank == 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            last_batch = batch
            if max_updates is not None and effective_step >= max_updates:
                stopped = True
                break
        cursor = 0
        if stopped:
            break
        if rank == 0:
            dropped_ids = [dataset.index.groups[index].group_id for index in sampler.dropped_indices]
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "epoch_complete",
                            "epoch": epoch,
                            "effective_step": effective_step,
                            "dropped_group_count": len(dropped_ids),
                            "dropped_group_ids": dropped_ids,
                            "drop_policy": "uniform_permutation_tail_without_duplicates",
                            "test_accessed": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if (epoch + 1) % int(config["validation"]["every_epochs"]) == 0 and rank == 0:
            _, metrics = evaluate_model(model, config, split="validation", device=device)
            primary, secondary = _selection(metrics)
            candidate = {"epoch": epoch, "effective_step": effective_step, "primary": primary,
                         "secondary": secondary, "metrics": metrics}
            if best is None or (primary, secondary) > (best["primary"], best["secondary"]):
                best = candidate
                save_checkpoint(
                    output_dir / "checkpoints" / "best_dev.pt", model=model, optimizer=optimizer,
                    config=config, epoch=epoch + 1, sampler_cursor=0, effective_step=effective_step,
                    best=best, provenance=provenance,
                )
                atomic_json_dump(best, output_dir / "best_dev_metrics.json")
            save_checkpoint(
                output_dir / "checkpoints" / "last.pt", model=model, optimizer=optimizer,
                config=config, epoch=epoch + 1, sampler_cursor=0,
                effective_step=effective_step, best=best, provenance=provenance,
            )
        if distributed:
            dist.barrier()
    if last_batch is not None and rank == 0:
        model.eval()
        with torch.no_grad():
            last_scores = _score(model, model.encode_pmgr_batch(last_batch), config)[0]
    checkpoint = None
    if rank == 0:
        checkpoint = save_checkpoint(
            output_dir / "checkpoints" / "last.pt", model=model, optimizer=optimizer,
            config=config, epoch=next_epoch, sampler_cursor=next_cursor, effective_step=effective_step,
            best=best, provenance=provenance,
        )
    if distributed:
        dist.barrier()
    reload_max_abs = None
    if rank == 0 and last_batch is not None and last_scores is not None:
        assert checkpoint is not None
        reloaded, _ = build_retriever(config, checkpoint=checkpoint["path"], device=device)
        reloaded.eval()
        with torch.no_grad():
            replay_scores = _score(reloaded, reloaded.encode_pmgr_batch(last_batch), config)[0]
        reload_max_abs = float((last_scores - replay_scores).abs().max())
        if reload_max_abs != 0.0:
            raise RuntimeError(f"checkpoint reload changed current-batch scores by {reload_max_abs}")
    summary = {
        "schema_version": 1,
        "status": "max_updates_reached" if stopped else "training_complete",
        "completion_label": "software_smoke" if max_updates is not None else "research_run_uninterpreted",
        "effective_steps": effective_step,
        "checkpoint": checkpoint,
        "checkpoint_reload_score_max_abs": reload_max_abs,
        "best_dev": best,
        "elapsed_seconds": time.time() - started,
    }
    if rank == 0:
        atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a PMGR/control objective")
    parser.add_argument("--config", required=True)
    parser.add_argument("--engine", choices=("direct", "two_level_replay"))
    parser.add_argument("--effective-groups", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--loss-mode")
    parser.add_argument("--rank-mix", type=float)
    parser.add_argument("--rank-eta", type=float)
    parser.add_argument("--distributed", choices=("none", "manual_sum_replay"))
    parser.add_argument("--video-encoder-microbatch", type=int)
    parser.add_argument("--text-encoder-microbatch", type=int)
    parser.add_argument("--score-video-block", type=int)
    parser.add_argument("--score-text-block", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    config = load_config(args.config, mode="train")
    config = resolved_with_overrides(
        config,
        engine__mode=args.engine,
        training__effective_groups=args.effective_groups,
        paths__output_dir=args.output_dir,
        loss__mode=args.loss_mode,
        loss__rank_mix=args.rank_mix,
        loss__rank_eta=args.rank_eta,
        engine__distributed=args.distributed,
        engine__video_encoder_microbatch=args.video_encoder_microbatch,
        engine__text_encoder_microbatch=args.text_encoder_microbatch,
        engine__score_video_block=args.score_video_block,
        engine__score_text_block=args.score_text_block,
    )
    manual_distributed = config["engine"]["distributed"] == "manual_sum_replay"
    if manual_distributed:
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            raise ValueError("manual_sum_replay must be launched with torchrun and at least 2 workers")
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if args.device.startswith("cuda") and torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend)
    else:
        device = torch.device(args.device)
    try:
        result = train(
            config, Path(config["paths"]["output_dir"]), device=device,
            max_updates=args.max_updates, resume=Path(args.resume) if args.resume else None,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
