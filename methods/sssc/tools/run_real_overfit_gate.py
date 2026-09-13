#!/usr/bin/env python3
"""Run a non-reportable fixed real-PH batch overfit gate for the corrected baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from method1.config import load_config
from method1.data import Method1Collator, Method1Dataset
from method1.distributed import DistributedRuntime
from method1.model_factory import build_upret_model
from method1.optimizer import build_upret_optimizer
from method1.trainer import Method1TrainModel, complete_optimizer_step
from method1.upstream import create_upret_tokenizer
from method1.utils import atomic_json_dump, sha256_json


class OverfitGateError(RuntimeError):
    pass


def _validate_gate_request(*, steps: int, batch_size: int) -> None:
    if steps < 2:
        raise OverfitGateError("steps must be at least two")
    if batch_size < 2:
        raise OverfitGateError("batch size must be at least two for contrastive overfit")
    if batch_size > 16:
        raise OverfitGateError("the non-reportable overfit gate is capped at 16 samples")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _objective(
    model: Method1TrainModel,
    batch: dict[str, Any],
    *,
    optimizer_step: int,
) -> torch.Tensor:
    return model(batch, optimizer_step=optimizer_step, microstep=0, epoch=0)["loss"]


def run_gate(
    *,
    config_path: Path,
    device_name: str,
    steps: int,
    batch_size: int,
    output: Path,
    upret_root: Path,
) -> dict[str, Any]:
    _validate_gate_request(steps=steps, batch_size=batch_size)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise OverfitGateError("CUDA was requested but is unavailable")
    config = load_config(config_path)
    if config.auxiliary.arm != "base_initial":
        raise OverfitGateError("the real overfit gate requires a base_initial config")
    tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
    dataset = Method1Dataset(config, split="train", tokenizer=tokenizer, augment=True)
    dataset.set_epoch(0)
    if len(dataset) < batch_size:
        raise OverfitGateError("training dataset is smaller than the requested fixed batch")
    batch = _move_batch(Method1Collator()([dataset[index] for index in range(batch_size)]), device)

    student, initialization = build_upret_model(config, upret_root=upret_root)
    student.to(device)
    runtime = DistributedRuntime()
    train_model = Method1TrainModel(
        student,
        config.auxiliary,
        runtime,
        baseline_seed=config.seed,
        inner_similarity_temperature=config.model.inner_similarity_temperature,
        checkpoint_score_blocks=False,
        video_pair_block=batch_size,
        text_pair_block=batch_size,
    ).to(device)
    optimizer = build_upret_optimizer(
        student, config, total_steps=steps, upret_root=upret_root
    ).optimizer
    parameter_name = "clip.visual.conv2_trans.weight"
    parameters = dict(student.named_parameters())
    if parameter_name not in parameters or not parameters[parameter_name].requires_grad:
        raise OverfitGateError(f"shared visual parameter is unavailable: {parameter_name}")
    visual_parameter = parameters[parameter_name]
    visual_before = visual_parameter.detach().clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    student.eval()
    with torch.no_grad():
        initial_probe = float(_objective(train_model, batch, optimizer_step=0))
    student.train()
    started = time.perf_counter()
    losses: list[float] = []
    gradient_norms: list[float] = []
    first_visual_gradient_norm: float | None = None
    for optimizer_step in range(steps):
        result = _objective(train_model, batch, optimizer_step=optimizer_step)
        result.backward()
        if optimizer_step == 0:
            if visual_parameter.grad is None:
                raise OverfitGateError("fixed batch produced no shared-visual gradient")
            first_visual_gradient_norm = float(visual_parameter.grad.detach().float().norm())
        gradient_norm = complete_optimizer_step(
            train_model,
            optimizer,
            max_grad_norm=config.training.max_grad_norm,
        )
        losses.append(float(result.detach()))
        gradient_norms.append(float(gradient_norm))
    elapsed = time.perf_counter() - started
    student.eval()
    with torch.no_grad():
        final_probe = float(_objective(train_model, batch, optimizer_step=0))
    visual_update_norm = float((visual_parameter.detach() - visual_before).float().norm())
    finite = bool(
        torch.isfinite(torch.tensor([*losses, *gradient_norms, initial_probe, final_probe])).all()
    )
    passed = bool(
        finite
        and first_visual_gradient_norm is not None
        and first_visual_gradient_norm > 0.0
        and visual_update_norm > 0.0
        and final_probe < initial_probe
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "role": "nonreportable_real_data_small_overfit_gate",
        "config": str(config_path.resolve()),
        "upret_root": str(upret_root.resolve()),
        "config_sha256": config.digest,
        "dataset": config.data.dataset,
        "split": "train",
        "seed": config.seed,
        "device": str(device),
        "batch_size": batch_size,
        "steps": steps,
        "video_uids": batch["video_uid"],
        "text_uids": batch["text_uid"],
        "batch_identity_sha256": sha256_json(
            list(zip(batch["video_uid"], batch["text_uid"], strict=True))
        ),
        "initial_probe_loss": initial_probe,
        "final_probe_loss": final_probe,
        "relative_probe_reduction": (initial_probe - final_probe) / max(abs(initial_probe), 1e-12),
        "training_losses": losses,
        "gradient_norms_before_outer_clip": gradient_norms,
        "shared_visual_parameter": parameter_name,
        "first_shared_visual_gradient_norm": first_visual_gradient_norm,
        "shared_visual_update_norm": visual_update_norm,
        "elapsed_wall_seconds": elapsed,
        "examples_per_second": batch_size * steps / max(elapsed, 1e-12),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "model_trainable_parameters": initialization["trainable_count"],
        "model_total_parameters": initialization["total_count"],
    }
    report["content_sha256"] = sha256_json(report)
    atomic_json_dump(report, output)
    if not passed:
        raise OverfitGateError(f"real-data overfit gate failed; inspect {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upret-root", type=Path, default=Path("third_party/UPRet"))
    args = parser.parse_args()
    report = run_gate(
        config_path=args.config,
        device_name=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        output=args.output,
        upret_root=args.upret_root,
    )
    print(args.output)
    print(report["content_sha256"])


if __name__ == "__main__":
    main()
