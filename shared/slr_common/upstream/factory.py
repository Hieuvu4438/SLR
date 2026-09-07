from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch


class CheckpointContractError(ValueError):
    pass


def _state_dict(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in value and isinstance(value[key], dict):
                value = value[key]
                break
    if not isinstance(value, dict) or not value:
        raise CheckpointContractError("checkpoint does not contain a model state dictionary")
    result = {}
    for key, tensor in value.items():
        normalized = key.removeprefix("module.")
        if torch.is_tensor(tensor):
            result[normalized] = tensor
    if not result:
        raise CheckpointContractError("checkpoint state dictionary contains no tensors")
    return result


def cico_task_config(config: dict[str, Any]) -> Namespace:
    data = config["data"]
    model = config["model"]
    return Namespace(
        local_rank=0,
        task_type="retrieval",
        sim_header=model.get("sim_header", "Filip"),
        loose_type=True,
        dual_mix=float(model.get("dual_mix", 0.5)),
        mix_design=model.get("mix_design", "balance"),
        feature_len=int(data["feature_len"]),
        max_words=int(data["max_words"]),
        alpha=float(data.get("alpha", 0.9)),
        aug_choose=model.get("aug_choose", "t2v"),
        visual_num_hidden_layers=int(model.get("visual_num_hidden_layers", 12)),
        cross_num_hidden_layers=int(model.get("cross_num_hidden_layers", 4)),
        linear_patch=model.get("linear_patch", "2d"),
        not_load_visual=False,
        pretrained_clip_name=model.get("pretrained_clip_name", "ViT-B/32"),
    )


def load_cico_core(
    config: dict[str, Any], checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> torch.nn.Module:
    """Instantiate CiCo from a complete local checkpoint without implicit downloads."""
    cico_root = Path(config["upstream"]["cico_root"]).resolve()
    if not cico_root.is_dir():
        raise CheckpointContractError(f"CiCo root does not exist: {cico_root}")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise CheckpointContractError(f"checkpoint does not exist: {checkpoint_path}")
    sys.path.insert(0, str(cico_root))
    try:
        state = _state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        return _load_cico_core_from_state(config, state, cico_root=cico_root, device=device)
    finally:
        try:
            sys.path.remove(str(cico_root))
        except ValueError:
            pass


def _load_cico_core_from_state(
    config: dict[str, Any],
    state: dict[str, torch.Tensor],
    *,
    cico_root: Path,
    device: str | torch.device,
) -> torch.nn.Module:
    modeling = importlib.import_module("modules.modeling")
    module_cross = importlib.import_module("modules.module_cross")
    clip_state = {
        key.removeprefix("clip."): value for key, value in state.items() if key.startswith("clip.")
    }
    required = {
        "visual.conv1.weight",
        "visual.positional_embedding",
        "visual.proj",
        "token_embedding.weight",
        "positional_embedding",
        "ln_final.weight",
        "text_projection",
    }
    missing_architecture = sorted(required - set(clip_state))
    if missing_architecture:
        raise CheckpointContractError(
            "checkpoint is not self-contained; missing CLIP architecture tensors: "
            + ", ".join(missing_architecture)
        )
    task = cico_task_config(config)
    cross_json = cico_root / "modules" / "cross-base" / "cross_config.json"
    cross_config = module_cross.CrossConfig(str(cross_json))
    core = modeling.CLIP4Clip(cross_config, clip_state.copy(), task)
    incompatible = core.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("cross.", "similarity_dense.", "frame_position_embeddings.")
    disallowed_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise CheckpointContractError(
            f"checkpoint incompatibility: missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
        )
    core.to(device)
    return core


def load_cico_tokenizer(config: dict[str, Any]):
    return load_cico_tokenizer_components(config)[0]


def load_cico_tokenizer_components(config: dict[str, Any]):
    cico_root = Path(config["upstream"]["cico_root"]).resolve()
    sys.path.insert(0, str(cico_root))
    try:
        module = importlib.import_module("modules.tokenization_clip")
        bpe = cico_root / "modules" / "bpe_simple_vocab_16e6.txt.gz"
        return module.SimpleTokenizer(str(bpe)), module.basic_clean, module.whitespace_clean, bpe
    finally:
        try:
            sys.path.remove(str(cico_root))
        except ValueError:
            pass
