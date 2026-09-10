"""Resumable train-only P14T I3D adaptation runner."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ocem.data.pseudoclips import PseudoClipDataset, load_pseudoclip_index
from ocem.provenance.hashes import canonical_json_sha256, sha256_file
from ocem.training.i3d_adaptation import (
    CicoAugmentationDraws,
    apply_cico_train_augmentation,
    build_cico_sgd,
    load_adaptation_i3d,
    sample_cico_augmentation_draws,
)


class AdaptationRunError(ValueError):
    """Raised when an adaptation run cannot satisfy its locked contract."""


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(records: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _core(model):
    return model.module if hasattr(model, "module") else model


def _worker_init(_worker_id: int) -> None:
    import cv2

    cv2.setNumThreads(0)


def _data_loader(dataset, *, batch_size: int, workers: int, shuffle_seed: int | None):
    import torch

    generator = None
    shuffle = shuffle_seed is not None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_worker_init if workers else None,
        drop_last=False,
    )


def eval_preprocess(rgb):
    """Apply the pinned CiCo validation center crop and normalization."""

    batch_size = rgb.shape[0]
    draws = CicoAugmentationDraws(
        horizontal_flip=False,
        color_jitter=(1.0, 1.0, 1.0),
        scale_uniform=np.full((batch_size, 1), 0.5),
        offset_uniform=np.full((batch_size, 2), 0.5),
    )
    return apply_cico_train_augmentation(rgb, draws)


def _train_epoch(
    *,
    model,
    optimizer,
    records: list[dict[str, Any]],
    epoch: int,
    seed: int,
    batch_size: int,
    workers: int,
    device,
    log_interval: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    dataset = PseudoClipDataset(records, training=True, seed=seed, epoch=epoch)
    loader = _data_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle_seed=seed + 1_000_003 * epoch,
    )
    model.train()
    loss_sum = 0.0
    correct = 0
    examples = 0
    started = time.monotonic()
    for batch_number, batch in enumerate(loader, start=1):
        raw = batch["rgb"].to(device, non_blocking=True)
        targets = batch["class"].to(device, non_blocking=True)
        draws = sample_cico_augmentation_draws(
            len(targets), seed=seed + 10_000_019 * epoch + batch_number
        )
        inputs = apply_cico_train_augmentation(raw, draws)
        optimizer.zero_grad(set_to_none=False)
        outputs = model(inputs)
        loss = functional.cross_entropy(outputs["logits"], targets, reduction="mean")
        if not bool(torch.isfinite(loss)):
            raise AdaptationRunError(
                f"nonfinite training loss at epoch {epoch}, batch {batch_number}"
            )
        loss.backward()
        gradients = [
            parameter.grad for parameter in _core(model).parameters() if parameter.grad is not None
        ]
        gradient_norm = torch.nn.utils.get_total_norm(gradients)
        if not bool(torch.isfinite(gradient_norm)):
            raise AdaptationRunError(
                f"nonfinite training gradient at epoch {epoch}, batch {batch_number}"
            )
        optimizer.step()
        examples += len(targets)
        loss_sum += float(loss.detach()) * len(targets)
        correct += int((outputs["logits"].detach().argmax(dim=1) == targets).sum())
        if batch_number % log_interval == 0 or batch_number == len(loader):
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"adapt train epoch={epoch} batch={batch_number}/{len(loader)} "
                f"examples={examples} loss={loss_sum / examples:.6f} "
                f"examples/s={examples / elapsed:.2f}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    return {
        "examples": examples,
        "batches": len(loader),
        "loss": loss_sum / examples,
        "top1": correct / examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-9),
    }


def _evaluate_holdout(
    *,
    model,
    records: list[dict[str, Any]],
    batch_size: int,
    workers: int,
    device,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    dataset = PseudoClipDataset(records, training=False)
    loader = _data_loader(dataset, batch_size=batch_size, workers=workers, shuffle_seed=None)
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            raw = batch["rgb"].to(device, non_blocking=True)
            targets = batch["class"].to(device, non_blocking=True)
            outputs = model(eval_preprocess(raw))
            loss = functional.cross_entropy(outputs["logits"], targets, reduction="mean")
            if not bool(torch.isfinite(loss)):
                raise AdaptationRunError("nonfinite holdout loss")
            examples += len(targets)
            loss_sum += float(loss) * len(targets)
            correct += int((outputs["logits"].argmax(dim=1) == targets).sum())
    elapsed = time.monotonic() - started
    return {
        "examples": examples,
        "batches": len(loader),
        "loss": loss_sum / examples,
        "top1": correct / examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-9),
        "used_for_checkpoint_selection": False,
    }


def _checkpoint_payload(
    *, model, optimizer, epoch: int, config_sha256: str, history: list[dict[str, Any]]
) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "schema_version": "ocem.p14t_i3d_adaptation_checkpoint.v1",
        "epoch": epoch,
        "arch": "InceptionI3d",
        "state_dict": _core(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_sha256": config_sha256,
        "history": history,
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_checkpoint(
    path: Path, *, model, optimizer, config_sha256: str
) -> tuple[int, list[dict[str, Any]]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise AdaptationRunError("adaptation checkpoint must be a mapping")
    if payload.get("schema_version") != "ocem.p14t_i3d_adaptation_checkpoint.v1":
        raise AdaptationRunError("unsupported adaptation checkpoint schema")
    if payload.get("config_sha256") != config_sha256:
        raise AdaptationRunError("adaptation checkpoint config digest mismatch")
    _core(model).load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_states" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
    epoch = int(payload.get("epoch", 0))
    history = payload.get("history")
    if epoch < 1 or not isinstance(history, list) or len(history) != epoch:
        raise AdaptationRunError("adaptation checkpoint epoch/history mismatch")
    return epoch, history


def run_p14t_i3d_adaptation(
    *,
    index: str | Path,
    expected_index_sha256: str,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    implementation: str | Path,
    expected_implementation_sha256: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    epochs: int = 15,
    batch_size: int = 4,
    workers: int = 4,
    seed: int = 0,
    snapshot_interval: int = 5,
    log_interval: int = 100,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the fixed 15-epoch recipe; the final epoch is selected a priori."""

    try:
        import torch
    except ImportError as error:
        raise AdaptationRunError("PyTorch is required for adaptation training") from error
    if (
        epochs != 15
        or batch_size != 4
        or workers < 0
        or seed != 0
        or snapshot_interval != 5
        or log_interval < 1
    ):
        raise AdaptationRunError(
            "locked P14T recipe requires epochs=15, batch_size=4, seed=0, snapshot_interval=5"
        )
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise AdaptationRunError("full P14T adaptation requires an available CUDA device")
    index, checkpoint, implementation = Path(index), Path(checkpoint), Path(implementation)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "checkpoint_latest.pt"
    report_path = output_dir / "run_report.json"
    metrics_path = output_dir / "epoch_metrics.jsonl"
    if not resume and latest_path.exists():
        raise AdaptationRunError(
            f"checkpoint already exists; pass resume=True or choose a new output directory: {latest_path}"
        )
    if resume and not latest_path.is_file():
        raise AdaptationRunError(f"resume checkpoint is missing: {latest_path}")

    train_records = load_pseudoclip_index(
        index, expected_sha256=expected_index_sha256, adaptation_split="train"
    )
    holdout_records = load_pseudoclip_index(
        index, expected_sha256=expected_index_sha256, adaptation_split="holdout"
    )
    config = {
        "schema_version": "ocem.p14t_i3d_adaptation_config.v1",
        "index_sha256": expected_index_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "implementation_sha256": expected_implementation_sha256,
        "device": str(resolved_device),
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "seed": seed,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "encoder_lr_coefficient": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "schedule_epochs": [20, 40],
        "schedule_gamma": 0.1,
        "effective_lr_schedule": "constant_0.01_for_all_15_epochs",
        "snapshot_interval": snapshot_interval,
        "checkpoint_selection": "fixed_final_epoch_15",
        "holdout_role": "monitor_only",
        "codec_roundtrip": False,
    }
    config_sha256 = canonical_json_sha256(config)
    model = load_adaptation_i3d(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        implementation=implementation,
        expected_implementation_sha256=expected_implementation_sha256,
        device=device,
    )
    gpu_index = resolved_device.index or 0
    model = torch.nn.DataParallel(model, device_ids=[gpu_index])
    optimizer = build_cico_sgd(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    start_epoch, history = 0, []
    if resume:
        start_epoch, history = _restore_checkpoint(
            latest_path, model=model, optimizer=optimizer, config_sha256=config_sha256
        )
    elif metrics_path.exists():
        raise AdaptationRunError(
            f"metrics already exist without a checkpoint: {metrics_path}; choose a new output directory"
        )

    report: dict[str, Any] = {
        "schema_version": "ocem.p14t_i3d_adaptation_run.v1",
        "status": "RUNNING",
        "config": config,
        "config_sha256": config_sha256,
        "inputs": {
            "index": str(index.resolve()),
            "index_sha256": sha256_file(index),
            "initial_checkpoint": str(checkpoint.resolve()),
            "initial_checkpoint_sha256": sha256_file(checkpoint),
            "implementation": str(implementation.resolve()),
            "implementation_sha256": sha256_file(implementation),
            "train_pseudo_clips": len(train_records),
            "holdout_pseudo_clips": len(holdout_records),
        },
        "output_dir": str(output_dir.resolve()),
        "start_epoch": start_epoch,
        "completed_epochs": start_epoch,
        "history": history,
        "data_policy": {
            "optimizer_source": "adaptation_split=train only",
            "holdout_used_for_optimizer": False,
            "holdout_used_for_checkpoint_selection": False,
            "validation_or_test_used": False,
        },
        "cuda_determinism_limit": (
            "torch 2.11 avg_pool3d_backward_cuda has no deterministic implementation; "
            "ordering and RNG are checkpointed, but bitwise rerun identity is not claimed"
        ),
    }
    _atomic_json(report, report_path)
    run_started = time.monotonic()
    try:
        for epoch_index in range(start_epoch, epochs):
            epoch = epoch_index + 1
            train_metrics = _train_epoch(
                model=model,
                optimizer=optimizer,
                records=train_records,
                epoch=epoch,
                seed=seed,
                batch_size=batch_size,
                workers=workers,
                device=resolved_device,
                log_interval=log_interval,
            )
            holdout_metrics = _evaluate_holdout(
                model=model,
                records=holdout_records,
                batch_size=batch_size,
                workers=workers,
                device=resolved_device,
            )
            epoch_record = {
                "epoch": epoch,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "train": train_metrics,
                "holdout": holdout_metrics,
            }
            history.append(epoch_record)
            checkpoint_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config_sha256=config_sha256,
                history=history,
            )
            _atomic_torch_save(checkpoint_payload, latest_path)
            _atomic_jsonl(history, metrics_path)
            if epoch % snapshot_interval == 0:
                _atomic_copy(latest_path, output_dir / f"checkpoint_epoch_{epoch:02d}.pt")
            report["completed_epochs"] = epoch
            report["history"] = history
            report["latest_checkpoint"] = {
                "path": str(latest_path.resolve()),
                "sha256": sha256_file(latest_path),
            }
            report["elapsed_seconds"] = time.monotonic() - run_started
            _atomic_json(report, report_path)
            print(
                f"adapt epoch={epoch}/{epochs} complete "
                f"train_loss={train_metrics['loss']:.6f} "
                f"holdout_loss={holdout_metrics['loss']:.6f}",
                flush=True,
            )
        final_path = output_dir / "checkpoint_final.pt"
        _atomic_copy(latest_path, final_path)
        report["status"] = "PASS"
        report["final_checkpoint"] = {
            "path": str(final_path.resolve()),
            "sha256": sha256_file(final_path),
            "fixed_epoch": epochs,
        }
        report["metrics_path"] = {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
        }
        report["elapsed_seconds"] = time.monotonic() - run_started
        report["ready_for_adapted_feature_extraction"] = True
        _atomic_json(report, report_path)
        return report
    except BaseException as error:
        report["status"] = "FAIL_TECHNICAL"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["elapsed_seconds"] = time.monotonic() - run_started
        _atomic_json(report, report_path)
        raise
