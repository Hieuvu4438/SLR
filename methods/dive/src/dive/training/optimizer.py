from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from dive.models.evidence import EvidenceEncoder


class OptimizerContractError(ValueError):
    """The evidence optimizer would violate ownership or decay policy."""


_NORMALIZATION_TYPES = (
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.instancenorm._InstanceNorm,
    nn.LocalResponseNorm,
)


@dataclass(frozen=True)
class OptimizerBundle:
    optimizer: torch.optim.AdamW
    manifest: dict[str, Any]


def _parameter_inventory(model: nn.Module) -> dict[int, tuple[str, nn.Module, str, nn.Parameter]]:
    inventory: dict[int, tuple[str, nn.Module, str, nn.Parameter]] = {}
    for module_name, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            parameter_id = id(parameter)
            if parameter_id in inventory:
                previous = inventory[parameter_id][0]
                raise OptimizerContractError(
                    f"parameter storage is registered under multiple names: {previous}, {full_name}"
                )
            inventory[parameter_id] = (full_name, module, local_name, parameter)
    return inventory


def build_evidence_optimizer(
    model: EvidenceEncoder,
    *,
    lr_projector: float = 1e-4,
    lr_pose: float = 1e-5,
    weight_decay: float = 0.01,
    forbidden_modules: Sequence[nn.Module] = (),
) -> OptimizerBundle:
    """Build AdamW with audited projector/pose ownership and exact no-decay rules.

    The two semantic ownership groups are expanded into decay/no-decay optimizer
    groups because AdamW applies weight decay at parameter-group granularity.
    """
    if lr_projector <= 0 or lr_pose <= 0 or weight_decay < 0:
        raise OptimizerContractError("optimizer learning rates/weight decay are invalid")
    model.enforce_frozen_batch_norm()
    inventory = _parameter_inventory(model)
    forbidden_ids = {
        id(parameter)
        for module in forbidden_modules
        for parameter in module.parameters()
    }
    buckets: dict[tuple[str, bool], list[tuple[str, nn.Parameter]]] = {
        ("projector", True): [],
        ("projector", False): [],
        ("pose", True): [],
        ("pose", False): [],
    }
    expected_ids: set[int] = set()
    for parameter_id, (name, module, local_name, parameter) in inventory.items():
        if not parameter.requires_grad:
            continue
        if parameter_id in forbidden_ids:
            raise OptimizerContractError(f"forbidden frozen parameter entered optimizer: {name}")
        if name.startswith("pose_encoder."):
            owner = "pose"
        elif name.startswith(("layer_norm.", "projector_in.", "projector_out.")):
            owner = "projector"
        else:
            raise OptimizerContractError(f"unrecognized trainable evidence parameter: {name}")
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            raise OptimizerContractError(f"batch-normalization parameter must be frozen: {name}")
        use_decay = local_name != "bias" and not isinstance(module, _NORMALIZATION_TYPES)
        buckets[(owner, use_decay)].append((name, parameter))
        expected_ids.add(parameter_id)

    if not expected_ids:
        raise OptimizerContractError("evidence model has no trainable parameters")
    parameter_groups: list[dict[str, Any]] = []
    manifest_groups: list[dict[str, Any]] = []
    assigned_ids: list[int] = []
    for owner, use_decay in (
        ("projector", True),
        ("projector", False),
        ("pose", True),
        ("pose", False),
    ):
        entries = buckets[(owner, use_decay)]
        if not entries:
            continue
        lr = lr_projector if owner == "projector" else lr_pose
        decay = weight_decay if use_decay else 0.0
        parameters = [parameter for _, parameter in entries]
        assigned_ids.extend(id(parameter) for parameter in parameters)
        parameter_groups.append(
            {
                "params": parameters,
                "lr": lr,
                "weight_decay": decay,
                "dive_owner": owner,
                "dive_decay": use_decay,
            }
        )
        manifest_groups.append(
            {
                "owner": owner,
                "decay": use_decay,
                "lr": lr,
                "weight_decay": decay,
                "parameters": [name for name, _ in entries],
                "numel": sum(parameter.numel() for parameter in parameters),
            }
        )
    if len(assigned_ids) != len(set(assigned_ids)) or set(assigned_ids) != expected_ids:
        raise OptimizerContractError("optimizer parameters are duplicated or incomplete")
    optimizer = torch.optim.AdamW(parameter_groups)
    manifest = {
        "schema_version": "dive_optimizer_manifest.v1",
        "optimizer": "adamw",
        "semantic_owners": ["projector", "pose"],
        "groups": manifest_groups,
        "trainable_parameter_count": len(expected_ids),
        "trainable_numel": sum(inventory[item][3].numel() for item in expected_ids),
    }
    return OptimizerBundle(optimizer=optimizer, manifest=manifest)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float = 0.1,
    minimum_lr_fraction: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise OptimizerContractError("scheduler total_steps must be positive")
    if not 0 < warmup_fraction <= 1:
        raise OptimizerContractError("scheduler warmup_fraction must be in (0,1]")
    if not 0 <= minimum_lr_fraction <= 1:
        raise OptimizerContractError("minimum_lr_fraction must be in [0,1]")
    warmup_steps = min(total_steps, max(1, math.ceil(total_steps * warmup_fraction)))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = total_steps - warmup_steps
        if decay_steps == 0:
            return minimum_lr_fraction
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_fraction + (1.0 - minimum_lr_fraction) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scheduler.dive_schedule = {  # type: ignore[attr-defined]
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "warmup_fraction": warmup_fraction,
        "minimum_lr_fraction": minimum_lr_fraction,
    }
    return scheduler


def optimizer_settings(config: Mapping[str, Any]) -> dict[str, float]:
    train = config["train"]
    return {
        "lr_projector": float(train["lr_projector"]),
        "lr_pose": float(train["lr_pose"]),
        "weight_decay": float(train["weight_decay"]),
    }
