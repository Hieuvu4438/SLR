from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from torch import nn


def build_inherited_bert_adam(model: nn.Module, config: dict[str, Any], total_steps: int):
    """Build the exact verified ``coef_lr=1`` CiCo optimizer groups."""
    if total_steps < 1:
        raise ValueError("optimizer needs at least one effective step")
    cico_root = Path(config["paths"]["cico_root"]).resolve()
    sys.path.insert(0, str(cico_root))
    try:
        BertAdam = importlib.import_module("modules.optimization").BertAdam
    finally:
        try:
            sys.path.remove(str(cico_root))
        except ValueError:
            pass
    train = config["training"]
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    no_decay_names = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    decay = [(name, parameter) for name, parameter in named if not any(item in name for item in no_decay_names)]
    no_decay = [(name, parameter) for name, parameter in named if any(item in name for item in no_decay_names)]
    decay_clip = [parameter for name, parameter in decay if "clip." in name]
    decay_other = [parameter for name, parameter in decay if "clip." not in name]
    no_decay_clip = [parameter for name, parameter in no_decay if "clip." in name]
    no_decay_other = [parameter for name, parameter in no_decay if "clip." not in name]
    lr = float(train["learning_rate"])
    weight_decay = float(train["weight_decay"])
    groups = [
        {"params": decay_clip, "weight_decay": weight_decay, "lr": lr},
        {"params": decay_other, "weight_decay": weight_decay},
        {"params": no_decay_clip, "weight_decay": 0.0, "lr": lr},
        {"params": no_decay_other, "weight_decay": 0.0},
    ]
    return BertAdam(
        groups,
        lr=lr,
        warmup=float(train["warmup_fraction"]),
        schedule="warmup_cosine",
        b1=float(train["beta1"]),
        b2=float(train["beta2"]),
        e=float(train["epsilon"]),
        t_total=total_steps,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
    )
