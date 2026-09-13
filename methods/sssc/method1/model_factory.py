from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .config import Method1Config, UPRET_COMMIT
from .utils import atomic_json_dump, sha256_file


class ModelLoadError(RuntimeError):
    pass


_ALLOWED_CLIP_MISMATCHES = {
    "clip.visual.conv1.weight",
    "clip.visual.positional_embedding",
}
_EXPECTED_NEW_CLIP_KEYS = {
    "clip.visual.conv2_trans.weight",
    "clip.logit_scale_first_softmax",
    "clip.logit_scale_sec_softmax",
}
_CLIP_METADATA_KEYS = {"input_resolution", "context_length", "vocab_size"}


def _activate_upret_import(upret_root: str | Path):
    root = Path(upret_root).resolve()
    actual = root / "modules" / "modeling.py"
    if not actual.is_file():
        raise ModelLoadError(f"pinned UPRet modeling source is missing: {actual}")
    existing = sys.modules.get("modules")
    if existing is not None:
        location = Path(getattr(existing, "__file__", "")).resolve()
        if root not in location.parents:
            raise ModelLoadError(
                f"top-level 'modules' is already imported from another project: {location}; "
                "construct UPRet in a clean process"
            )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from modules.modeling import CLIP4Clip
    from modules.module_cross import CrossConfig

    return CLIP4Clip, CrossConfig


def _load_clip_state(path: str | Path) -> dict[str, torch.Tensor]:
    source = Path(path)
    if not source.is_file():
        raise ModelLoadError(f"CLIP checkpoint is missing: {source}")
    try:
        state = torch.jit.load(str(source), map_location="cpu").eval().state_dict()
    except RuntimeError:
        state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ModelLoadError("CLIP checkpoint did not contain a state dictionary")
    return state


def config_to_upret(config: Method1Config) -> SimpleNamespace:
    return SimpleNamespace(
        local_rank=0,
        rank=0,
        world_size=1,
        max_words=config.data.text_max_positions,
        max_frames=config.data.feature_len,
        feature_len=config.data.feature_len,
        alpha=config.data.agnostic_weight,
        dual_mix=config.model.dual_mix,
        mix_design=config.model.mix_design,
        visual_num_hidden_layers=config.model.visual_layers,
        text_num_hidden_layers=config.model.text_layers,
        linear_patch=config.model.linear_patch,
        sim_header=config.model.sim_header,
        loose_type=config.model.loose_type,
        freeze_layer_num=config.model.freeze_layer_num,
        coef_lr=1.0,
        aug_choose="t2v",
        not_load_visual=False,
        cross_model="cross-base",
        cross_num_hidden_layers=4,
        pretrained_clip_name=str(Path(config.model.clip_checkpoint_path).resolve()),
    )


def _seed_initialization(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _initialize_clip_weights(
    model: torch.nn.Module,
    clip_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    target = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    matched: list[str] = []
    mismatched: dict[str, dict[str, list[int]]] = {}
    unexpected: list[str] = []
    for source_key, value in clip_state.items():
        if source_key in _CLIP_METADATA_KEYS:
            continue
        target_key = f"clip.{source_key}"
        if target_key not in target:
            unexpected.append(target_key)
            continue
        if target[target_key].shape != value.shape:
            mismatched[target_key] = {
                "checkpoint": list(value.shape),
                "model": list(target[target_key].shape),
            }
            continue
        loadable[target_key] = value.float() if target[target_key].is_floating_point() else value
        matched.append(target_key)
    illegal_mismatches = sorted(set(mismatched) - _ALLOWED_CLIP_MISMATCHES)
    if illegal_mismatches:
        raise ModelLoadError(f"unexpected transferable CLIP shape mismatches: {illegal_mismatches}")
    if unexpected:
        raise ModelLoadError(f"unexpected CLIP checkpoint keys: {unexpected[:10]}")
    incompatible = model.load_state_dict(loadable, strict=False)
    if incompatible.unexpected_keys:
        raise ModelLoadError(f"unexpected initialized keys: {incompatible.unexpected_keys}")
    missing_clip = {
        key for key in incompatible.missing_keys if key.startswith("clip.")
    }
    expected_missing = _ALLOWED_CLIP_MISMATCHES | _EXPECTED_NEW_CLIP_KEYS
    illegal_missing = sorted(missing_clip - expected_missing)
    if illegal_missing:
        raise ModelLoadError(f"transferable CLIP parameters were not loaded: {illegal_missing}")
    return {
        "matched": sorted(matched),
        "matched_count": len(matched),
        "mismatched": mismatched,
        "expected_new_clip_keys": sorted(_EXPECTED_NEW_CLIP_KEYS & missing_clip),
        "new_baseline_parameters": sorted(
            key for key in incompatible.missing_keys if not key.startswith("clip.")
        ),
    }


def apply_upret_trainability(model: torch.nn.Module, freeze_layer_num: int) -> dict[str, bool]:
    if not 0 <= freeze_layer_num <= 12:
        raise ModelLoadError("freeze_layer_num must be in [0,12]")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.clip.named_parameters():
        keep_trainable = (
            name.startswith("ln_final.")
            or name.startswith("text_projection")
            or name.startswith("logit_scale")
            or name.startswith("visual.")
            or name.startswith("seq")
            or name.startswith("cross_atten")
        )
        if name.startswith("transformer.resblocks."):
            layer = int(name.split(".resblocks.", 1)[1].split(".", 1)[0])
            keep_trainable = layer >= freeze_layer_num
        if not keep_trainable:
            parameter.requires_grad_(False)
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def build_upret_model(
    config: Method1Config,
    *,
    upret_root: str | Path = "third_party/UPRet",
    report_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    root = Path(upret_root).resolve()
    try:
        revision = (
            __import__("subprocess")
            .run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception as error:
        raise ModelLoadError(f"cannot verify UPRet checkout revision: {error}") from error
    if revision != UPRET_COMMIT:
        raise ModelLoadError(f"UPRet checkout is {revision}, expected {UPRET_COMMIT}")
    CLIP4Clip, CrossConfig = _activate_upret_import(root)
    args = config_to_upret(config)
    _seed_initialization(config.seed)
    clip_state = _load_clip_state(config.model.clip_checkpoint_path)
    cross_config, _ = CrossConfig.get_config(
        "cross-base", cache_dir=None, type_vocab_size=2, state_dict=None, task_config=args
    )
    if cross_config is None:
        raise ModelLoadError("UPRet cross-base configuration could not be loaded")
    model = CLIP4Clip(cross_config, clip_state.copy(), task_config=args).float()
    load_report = _initialize_clip_weights(model, clip_state)
    model.sample_num = config.model.distribution_samples
    model.eps = config.model.sample_ot_epsilon
    model.max_iter = config.model.sample_ot_max_iterations
    model.ot_weight = config.model.sample_ot_logit_weight
    trainable = apply_upret_trainability(model, config.model.freeze_layer_num)
    actual_text_layers = len(model.clip.transformer.resblocks)
    if actual_text_layers != config.model.text_layers:
        raise ModelLoadError(
            f"initialized CLIP has {actual_text_layers} text layers, expected {config.model.text_layers}"
        )
    report = {
        "schema_version": 1,
        "status": "ready",
        "upret_commit": revision,
        "clip_checkpoint": str(Path(config.model.clip_checkpoint_path).resolve()),
        "clip_sha256": sha256_file(config.model.clip_checkpoint_path),
        "load": load_report,
        "architecture": {
            "feature_len": model.clip.visual.feature_len,
            "embedding_dim": int(model.clip.text_projection.shape[1]),
            "text_layers": actual_text_layers,
            "distribution_samples": model.sample_num,
            "sample_ot_epsilon": model.eps,
            "sample_ot_max_iterations": model.max_iter,
            "sample_ot_logit_weight": model.ot_weight,
        },
        "trainable_parameters": trainable,
        "trainable_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if report_path is not None:
        atomic_json_dump(report, report_path)
    return model, report


def load_exact_student_state(model: torch.nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = value.get("student_state_dict") if isinstance(value, dict) else None
    if not isinstance(state, dict):
        raise ModelLoadError("trained checkpoint must contain student_state_dict")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ModelLoadError(f"trained student state is not exact: {error}") from error
    return value
