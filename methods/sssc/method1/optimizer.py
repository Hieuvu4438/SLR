from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import Method1Config


class OptimizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizerBuild:
    optimizer: torch.optim.Optimizer
    parameter_groups: tuple[dict[str, Any], ...]


def _load_pinned_bert_adam(upret_root: str | Path):
    source = Path(upret_root).resolve() / "modules" / "optimization.py"
    if not source.is_file():
        raise OptimizerError(f"pinned UPRet optimizer source is missing: {source}")
    spec = importlib.util.spec_from_file_location("_method1_upret_optimization", source)
    if spec is None or spec.loader is None:
        raise OptimizerError(f"cannot load UPRet optimizer source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BertAdam


def build_upret_optimizer(
    model: torch.nn.Module,
    config: Method1Config,
    *,
    total_steps: int,
    upret_root: str | Path = "third_party/UPRet",
    bert_adam_cls: type[torch.optim.Optimizer] | None = None,
) -> OptimizerBuild:
    """Reproduce UPRet's coef_lr=1 four-group BertAdam construction."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if config.training.optimizer != "upstream_bertadam":
        raise OptimizerError("only the pinned upstream BertAdam is permitted")
    if config.training.schedule != "upstream_warmup_cosine":
        raise OptimizerError("only UPRet's internal warmup_cosine schedule is permitted")

    named = list(model.named_parameters())
    if len({name for name, _ in named}) != len(named):
        raise OptimizerError("model has duplicate parameter names")
    no_decay_markers = ("bias", "LayerNorm.bias", "LayerNorm.weight")

    def is_no_decay(name: str) -> bool:
        return any(marker in name for marker in no_decay_markers)

    definitions = (
        ("decay_clip", lambda name: not is_no_decay(name) and "clip." in name, config.training.weight_decay),
        ("decay_nonclip", lambda name: not is_no_decay(name) and "clip." not in name, config.training.weight_decay),
        ("no_decay_clip", lambda name: is_no_decay(name) and "clip." in name, 0.0),
        ("no_decay_nonclip", lambda name: is_no_decay(name) and "clip." not in name, 0.0),
    )
    reports: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for label, predicate, decay in definitions:
        selected = [(name, parameter) for name, parameter in named if predicate(name)]
        names = [name for name, _ in selected]
        overlap = assigned.intersection(names)
        if overlap:
            raise OptimizerError(f"parameters assigned to multiple groups: {sorted(overlap)}")
        assigned.update(names)
        grouped.append(
            {
                "params": [parameter for _, parameter in selected],
                "weight_decay": decay,
                "lr": config.training.learning_rate,
            }
        )
        reports.append(
            {
                "name": label,
                "parameter_names": names,
                "parameter_count": sum(parameter.numel() for _, parameter in selected),
                "trainable_parameter_count": sum(
                    parameter.numel() for _, parameter in selected if parameter.requires_grad
                ),
                "weight_decay": decay,
                "learning_rate": config.training.learning_rate,
            }
        )
    missing = sorted(set(dict(named)) - assigned)
    if missing:
        raise OptimizerError(f"parameters were not assigned to an optimizer group: {missing}")

    optimizer_type = bert_adam_cls or _load_pinned_bert_adam(upret_root)
    optimizer = optimizer_type(
        grouped,
        lr=config.training.learning_rate,
        warmup=config.training.warmup_fraction,
        schedule="warmup_cosine",
        b1=config.training.betas[0],
        b2=config.training.betas[1],
        e=config.training.epsilon,
        t_total=total_steps,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
    )
    return OptimizerBuild(optimizer=optimizer, parameter_groups=tuple(reports))
