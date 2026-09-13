from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .auxiliary_cache import Method1AuxiliaryCache
from .checkpoints import (
    DevSelection,
    atomic_torch_save,
    capture_rng_state,
    make_training_checkpoint,
    restore_rng_state,
    validate_resume_identity,
)
from .config import Method1Config
from .data import Method1Collator, Method1Dataset, TrimmedDistributedGroupSampler
from .distributed import DistributedRuntime
from .inference import evaluate_loaded_student
from .model_factory import build_upret_model, load_exact_student_state
from .optimizer import build_upret_optimizer
from .provenance import runtime_environment_report
from .reference_pipeline import reference_cache_identity
from .trainer import Method1TrainModel, complete_optimizer_step
from .upstream import create_upret_tokenizer
from .utils import atomic_json_dump, sha256_file, sha256_json


class TrainingError(RuntimeError):
    pass


def _implementation_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--", "methods/sssc"], check=False
    ).returncode
    return revision + ("+dirty" if dirty else "")


def _initialize_runtime(requested_device: str) -> tuple[DistributedRuntime, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    owned_process_group = False
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and requested_device != "cpu" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        owned_process_group = True
    runtime = DistributedRuntime.current()
    if (runtime.rank, runtime.world_size) != (rank, world_size):
        raise TrainingError("torchrun environment and initialized process group disagree")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise TrainingError("CUDA was requested but is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
    return runtime, device, owned_process_group


def _artifact_hashes(config: Method1Config, *, method_stage: bool) -> dict[str, Any]:
    manifest = Path(config.data.manifest_dir)
    hashes: dict[str, Any] = {
        "manifests": {
            name: sha256_file(manifest / name)
            for name in ("texts.jsonl", "videos.jsonl", "groups.jsonl", "splits.json", "resources.json")
        },
        "tokenizer": sha256_file(config.model.bpe_path),
        "clip_initialization": sha256_file(config.model.clip_checkpoint_path),
        "teacher": None,
        "reference_cache": None,
        "mining_cache": None,
        "negative_span_cache": None,
    }
    if method_stage:
        identity = reference_cache_identity(config)
        cache_root = Path(config.reference.cache_dir)
        cache_meta = json.loads((cache_root / "cache_meta.json").read_text(encoding="utf-8"))
        mining_meta = json.loads(
            (cache_root / "auxiliary" / "mining" / "mining_report.json").read_text(
                encoding="utf-8"
            )
        )
        negative_meta = json.loads(
            (
                cache_root
                / "auxiliary"
                / "negative_spans"
                / "negative_cache_meta.json"
            ).read_text(encoding="utf-8")
        )
        hashes.update(
            {
                "teacher": identity.teacher_checkpoint_sha256,
                "reference_cache": cache_meta["content_sha256"],
                "mining_cache": mining_meta["content_sha256"],
                "negative_span_cache": negative_meta["content_sha256"],
            }
        )
    return hashes


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "candidate_counts": metrics.get("candidate_counts"),
        "ties": metrics.get("ties"),
        "V2T": {
            key: value for key, value in metrics.get("V2T", {}).items() if key != "ranks"
        },
        "T2V": {
            key: value for key, value in metrics.get("T2V", {}).items() if key != "ranks"
        },
    }


def _collect_rng_states(runtime: DistributedRuntime) -> list[dict[str, Any]]:
    local_state = capture_rng_state()
    if runtime.world_size == 1:
        return [local_state]
    states: list[dict[str, Any] | None] = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise TrainingError("failed to gather per-rank RNG states")
    return [state for state in states if state is not None]


def _local_batch_identity(batch: dict[str, Any]) -> list[Any]:
    if "edit_uids" in batch:
        return list(zip(batch["video_uid"], batch["text_uid"], batch["edit_uids"]))
    return list(zip(batch["video_uid"], batch["text_uid"]))


def _batch_identity_hash(batch: dict[str, Any], runtime: DistributedRuntime) -> str:
    local = _local_batch_identity(batch)
    if runtime.world_size == 1:
        return sha256_json(local)
    gathered: list[Any] = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(gathered, local)
    return sha256_json(gathered)


def train_stage(
    config: Method1Config,
    *,
    stage: str,
    max_steps: int | None = None,
    resume: str | Path | None = None,
    requested_device: str = "auto",
    upret_root: str | Path = "third_party/UPRet",
) -> dict[str, Any]:
    if stage not in {"base", "method"}:
        raise ValueError("training stage must be base or method")
    method_stage = stage == "method"
    if not method_stage and config.auxiliary.arm != "base_initial":
        raise TrainingError("train-base requires auxiliary.arm=base_initial")
    if method_stage and config.auxiliary.arm not in {
        "base_continuation",
        "span_independent",
        "span_shared",
        "span_random_support",
        "caption_hn",
        "fsc_local",
        "fsc_local_caption_hn",
    }:
        raise TrainingError("train-method received an unsupported or deferred arm")
    if config.training.mixed_precision != "none":
        raise TrainingError("the correctness-reference trainer currently requires float32")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if (
        method_stage
        and config.auxiliary.negatives_per_caption == 1
        and max_steps is not None
        and max_steps > 200
    ):
        raise TrainingError("the K=1 engineering pilot is capped at 200 optimizer steps")

    runtime, device, owned_process_group = _initialize_runtime(requested_device)
    try:
        if config.training.global_contrastive_batch % runtime.world_size:
            raise TrainingError("global contrastive batch must be divisible by world size")
        local_batch = config.training.global_contrastive_batch // runtime.world_size
        output_root = Path(config.output.root)
        report_path = output_root / "model_initialization.json" if runtime.rank == 0 else None
        student, model_report = build_upret_model(
            config, upret_root=upret_root, report_path=report_path
        )
        if method_stage:
            baseline_checkpoint = load_exact_student_state(student, config.reference.checkpoint)
            if (
                baseline_checkpoint.get("arm") != "base_initial"
                or baseline_checkpoint.get("dev_selection") is None
                or not baseline_checkpoint.get("training_run_complete", False)
            ):
                raise TrainingError("every method arm must branch from the selected base_initial checkpoint")
        student = student.to(device)
        tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
        auxiliary_cache = None
        if method_stage and config.auxiliary.arm != "base_continuation":
            auxiliary_cache = Method1AuxiliaryCache(
                config.reference.cache_dir,
                expected_identity=reference_cache_identity(config),
            )
        dataset = Method1Dataset(
            config,
            split="train",
            tokenizer=tokenizer,
            augment=True,
            auxiliary_cache=auxiliary_cache,
        )
        sampler = TrimmedDistributedGroupSampler(
            len(dataset),
            seed=config.seed,
            global_batch_size=config.training.global_contrastive_batch,
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=local_batch,
            sampler=sampler,
            drop_last=True,
            num_workers=config.training.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=Method1Collator(),
            persistent_workers=False,
        )
        steps_per_epoch = len(loader)
        epochs = config.training.finetune_epochs if method_stage else config.training.base_epochs
        planned_steps = steps_per_epoch * epochs
        optimizer_steps = min(planned_steps, max_steps) if max_steps is not None else planned_steps
        optimizer_build = build_upret_optimizer(
            student,
            config,
            total_steps=optimizer_steps,
            upret_root=upret_root,
        )
        optimizer = optimizer_build.optimizer
        artifacts = _artifact_hashes(config, method_stage=method_stage)
        artifacts["training_protocol"] = {
            "stage": stage,
            "planned_steps": planned_steps,
            "effective_step_budget": optimizer_steps,
            "world_size": runtime.world_size,
            "local_batch": local_batch,
        }
        start_epoch = 0
        next_batch_index = 0
        global_step = 0
        if resume is not None:
            resumed = torch.load(resume, map_location="cpu", weights_only=True)
            validate_resume_identity(resumed, config=config, artifact_hashes=artifacts)
            student.load_state_dict(resumed["student_state_dict"], strict=True)
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            rng_states = resumed.get("rng_state_by_rank")
            if rng_states is not None:
                if len(rng_states) != runtime.world_size:
                    raise TrainingError("resume checkpoint RNG world size differs")
                restore_rng_state(rng_states[runtime.rank])
            else:
                restore_rng_state(resumed["rng_state"])
            start_epoch = int(resumed["epoch"])
            next_batch_index = int(resumed["next_batch_index"])
            global_step = int(resumed["global_step"])
            if global_step > optimizer_steps:
                raise TrainingError("resume checkpoint is beyond the configured step budget")
        train_model = Method1TrainModel(
            student,
            config.auxiliary,
            runtime,
            baseline_seed=config.seed,
            inner_similarity_temperature=config.model.inner_similarity_temperature,
            checkpoint_score_blocks=config.training.checkpoint_deterministic_score_blocks,
            video_pair_block=config.training.train_video_pair_block,
            text_pair_block=config.training.train_text_pair_block,
        ).to(device)
        visible_model: torch.nn.Module = train_model
        if runtime.world_size > 1:
            visible_model = DistributedDataParallel(
                train_model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        optimizer.zero_grad()
        output_root.mkdir(parents=True, exist_ok=True)
        if runtime.rank == 0:
            environment_path = output_root / "environment.json"
            atomic_json_dump(runtime_environment_report(), environment_path)
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "resolved_config": {key: value for key, value in asdict(config).items() if key != "source_path"},
                    "config_sha256": config.digest,
                    "artifact_hashes": artifacts,
                    "environment": {
                        "path": str(environment_path.resolve()),
                        "sha256": sha256_file(environment_path),
                    },
                    "optimizer_groups": list(optimizer_build.parameter_groups),
                    "planned_steps": planned_steps,
                    "effective_step_budget": optimizer_steps,
                    "local_batch": local_batch,
                    "world_size": runtime.world_size,
                    "dropped_tail": sampler.dropped_tail,
                    "model_counts": {
                        "trainable": model_report["trainable_count"],
                        "total": model_report["total_count"],
                    },
                },
                output_root / "run_manifest.json",
            )
        best_selection: DevSelection | None = None
        best_path = output_root / "best_dev.pt"
        if resume is not None and best_path.is_file():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
            if best_payload.get("dev_metrics") is not None:
                best_selection = DevSelection.from_metrics(
                    best_payload["dev_metrics"], int(best_payload["global_step"])
                )
        stopped = False
        final_metrics: dict[str, Any] | None = None
        final_next_batch = next_batch_index
        implementation_revision = _implementation_revision()
        if global_step == optimizer_steps:
            stopped = True
        for epoch in range(start_epoch, epochs):
            if stopped:
                break
            dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            permutation_hash = sha256_json(sampler.permutation().tolist())
            for batch_index, batch in enumerate(loader):
                if epoch == start_epoch and batch_index < next_batch_index:
                    continue
                result = visible_model(
                    _move_batch(batch, device),
                    optimizer_step=global_step,
                    microstep=0,
                    epoch=epoch,
                )
                result["loss"].backward()
                gradient_norm = complete_optimizer_step(
                    visible_model,
                    optimizer,
                    max_grad_norm=config.training.max_grad_norm,
                )
                global_step += 1
                final_next_batch = batch_index + 1
                debug_batch_hash = (
                    _batch_identity_hash(batch, runtime)
                    if config.training.deterministic_debug
                    else None
                )
                if runtime.rank == 0 and debug_batch_hash is not None:
                    atomic_json_dump(
                        {
                            "schema_version": 1,
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "global_step": global_step,
                            "batch_identity_sha256": debug_batch_hash,
                        },
                        output_root / "debug_batches" / f"step_{global_step:08d}.json",
                    )
                if runtime.rank == 0 and (
                    global_step == 1 or global_step % 10 == 0 or global_step == optimizer_steps
                ):
                    atomic_json_dump(
                        {
                            "schema_version": 1,
                            "epoch": epoch,
                            "next_batch_index": final_next_batch,
                            "global_step": global_step,
                            "loss": float(result["loss"].detach()),
                            "base": float(result["base"]),
                            "auxiliary": float(result.get("auxiliary", torch.tensor(0.0))),
                            "caption_auxiliary": float(
                                result.get("caption_auxiliary", torch.tensor(0.0))
                            ),
                            "fsc_auxiliary": float(
                                result.get("fsc_auxiliary", torch.tensor(0.0))
                            ),
                            "negative_encoding_slots": int(
                                result.get("negative_encoding_slots", torch.tensor(0))
                            ),
                            "valid_negative_count": int(
                                result.get("valid_negative_count", torch.tensor(0))
                            ),
                            "gradient_norm": float(gradient_norm),
                            "batch_identity_sha256": debug_batch_hash
                            or sha256_json(_local_batch_identity(batch)),
                        },
                        output_root / "latest_train_step.json",
                    )
                if (
                    config.output.save_last
                    and global_step % config.training.checkpoint_every_steps == 0
                ):
                    rng_states = _collect_rng_states(runtime)
                    if runtime.rank == 0:
                        next_epoch = epoch + 1 if final_next_batch >= steps_per_epoch else epoch
                        resumed_batch = 0 if next_epoch == epoch + 1 else final_next_batch
                        periodic = make_training_checkpoint(
                            config=config,
                            student=student,
                            optimizer=optimizer,
                            epoch=next_epoch,
                            next_batch_index=resumed_batch,
                            global_step=global_step,
                            sampler_state={
                                "seed": config.seed,
                                "epoch": epoch,
                                "permutation_sha256": permutation_hash,
                                "global_batch_size": config.training.global_contrastive_batch,
                                "dropped_tail": sampler.dropped_tail,
                            },
                            artifact_hashes=artifacts,
                            implementation_revision=implementation_revision,
                            dev_metrics=None,
                            rng_state=rng_states[0],
                        )
                        periodic["rng_state_by_rank"] = rng_states
                        periodic["training_run_complete"] = False
                        atomic_torch_save(periodic, output_root / "last.pt")
                if global_step >= optimizer_steps:
                    stopped = True
                    break

            if runtime.world_size > 1:
                dist.barrier()
            rng_states = _collect_rng_states(runtime)
            if runtime.rank == 0:
                final_metrics = evaluate_loaded_student(
                    config,
                    student,
                    split="dev",
                    tokenizer=tokenizer,
                    device=device,
                )
                selection = DevSelection.from_metrics(final_metrics, global_step)
                checkpoint_epoch = epoch + 1 if final_next_batch >= steps_per_epoch else epoch
                checkpoint_next_batch = 0 if checkpoint_epoch == epoch + 1 else final_next_batch
                checkpoint = make_training_checkpoint(
                    config=config,
                    student=student,
                    optimizer=optimizer,
                    epoch=checkpoint_epoch,
                    next_batch_index=checkpoint_next_batch,
                    global_step=global_step,
                    sampler_state={
                        "seed": config.seed,
                        "epoch": epoch,
                        "permutation_sha256": permutation_hash,
                        "global_batch_size": config.training.global_contrastive_batch,
                        "dropped_tail": sampler.dropped_tail,
                    },
                    artifact_hashes=artifacts,
                    implementation_revision=implementation_revision,
                    dev_metrics=final_metrics,
                    rng_state=rng_states[0],
                )
                checkpoint["rng_state_by_rank"] = rng_states
                checkpoint["training_run_complete"] = False
                if config.output.save_last:
                    atomic_torch_save(checkpoint, output_root / "last.pt")
                if config.output.save_best_dev and selection.beats(best_selection):
                    atomic_torch_save(checkpoint, best_path)
                    best_selection = selection
                atomic_json_dump(final_metrics, output_root / f"dev_step_{global_step}.json")
            if runtime.world_size > 1:
                dist.barrier()
            if stopped:
                break
            next_batch_index = 0
        full_protocol_complete = global_step == planned_steps == optimizer_steps
        if runtime.world_size > 1:
            dist.barrier()
        if runtime.rank == 0 and full_protocol_complete:
            for selected_path in (best_path, output_root / "last.pt"):
                if not selected_path.is_file():
                    continue
                selected_checkpoint = torch.load(
                    selected_path, map_location="cpu", weights_only=True
                )
                selected_checkpoint["training_run_complete"] = True
                selected_checkpoint["completed_global_step"] = global_step
                selected_checkpoint["completed_protocol_steps"] = planned_steps
                atomic_torch_save(selected_checkpoint, selected_path)
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "global_step": global_step,
                    "planned_steps": planned_steps,
                    "best_dev": str(best_path.resolve()),
                },
                output_root / "training_complete.json",
            )
        if runtime.world_size > 1:
            dist.barrier()
        return {
            "schema_version": 1,
            "status": "complete" if full_protocol_complete else "pilot_complete",
            "stage": stage,
            "arm": config.auxiliary.arm,
            "global_step": global_step,
            "effective_step_budget": optimizer_steps,
            "full_protocol_complete": full_protocol_complete,
            "output_root": str(output_root.resolve()),
            "best_dev": str(best_path.resolve()) if best_path.is_file() else None,
            "dev_metrics": _compact_metrics(final_metrics) if runtime.rank == 0 else None,
            "rank": runtime.rank,
            "world_size": runtime.world_size,
        }
    finally:
        if owned_process_group and dist.is_initialized():
            dist.destroy_process_group()
