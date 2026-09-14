from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from pmgr.scoring import EncodedPMGRBatch, text_valid_from_cico, video_valid_from_legacy
from slr_common.upstream.factory import (
    CheckpointContractError,
    _load_cico_core_from_state,
    _state_dict,
    load_cico_tokenizer_components,
)
from slr_common.utils import sha256_file


class PMGRRetriever(nn.Module):
    """Parameter-preserving adapter around the pinned CiCo model."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    @property
    def logit_scale(self) -> nn.Parameter:
        return self.core.clip.logit_scale

    def encode_video(self, video_features: Tensor, video_padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        if video_features.ndim != 4 or video_features.shape[-1] != 1:
            raise ValueError("CiCo video input must have [videos,1024,features,1]")
        result = self.core.get_visual_output(
            video_features,
            video_padding_mask,
            shaped=True,
            video_frame=1,
            get_hidden=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("unexpected CiCo video encoder contract")
        returned_mask, hidden, _ = result
        if hidden.ndim != 3 or hidden.shape[0] != video_features.shape[0]:
            raise RuntimeError("CiCo video encoder squeezed or reordered the batch")
        valid = video_valid_from_legacy(returned_mask, hidden_length=hidden.shape[1])
        return hidden, valid

    def encode_text(self, input_ids: Tensor, segment_ids: Tensor, input_mask: Tensor) -> tuple[Tensor, Tensor]:
        result = self.core.get_sequence_output(
            input_ids,
            segment_ids,
            input_mask,
            shaped=True,
            get_hidden=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("unexpected CiCo text encoder contract")
        returned_mask, hidden, _ = result
        if hidden.ndim != 3 or hidden.shape[0] != input_ids.shape[0]:
            raise RuntimeError("CiCo text encoder squeezed or reordered the batch")
        valid = text_valid_from_cico(returned_mask, hidden_length=hidden.shape[1])
        return hidden, valid

    def encode_pmgr_batch(self, batch: dict[str, Any]) -> EncodedPMGRBatch:
        video_hidden, video_valid = self.encode_video(
            batch["video_features"], batch["video_padding_mask"]
        )
        text_hidden, text_valid = self.encode_text(
            batch["input_ids"], batch["segment_ids"], batch["input_mask"]
        )
        aug_hidden, aug_valid = self.encode_text(
            batch["aug_input_ids"], batch["aug_segment_ids"], batch["aug_input_mask"]
        )
        encoded = EncodedPMGRBatch(
            video_hidden=video_hidden,
            video_valid=video_valid,
            text_hidden=text_hidden,
            text_valid=text_valid,
            aug_hidden=aug_hidden,
            aug_valid=aug_valid,
        )
        encoded.validate()
        return encoded


def _baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["paths"]["baseline_resolved_args"])
    expected_hash = config["paths"]["baseline_resolved_args_sha256"]
    if sha256_file(path) != expected_hash:
        raise CheckpointContractError("baseline resolved-argument SHA-256 mismatch")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CheckpointContractError(f"cannot load baseline resolved arguments: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointContractError("baseline resolved arguments must be a mapping")
    actual_commit = value.get("upstream", {}).get("cico_commit")
    if actual_commit != config["upstream_commit"]:
        raise CheckpointContractError(
            f"baseline points at CiCo {actual_commit}, expected {config['upstream_commit']}"
        )
    value["upstream"]["cico_root"] = config["paths"]["cico_root"]
    return value


def _core_state(raw: Any) -> dict[str, Tensor]:
    state = _state_dict(raw)
    if any(key.startswith("core.") for key in state):
        core = {
            key.removeprefix("core."): tensor
            for key, tensor in state.items()
            if key.startswith("core.")
        }
        if not core:
            raise CheckpointContractError("wrapper checkpoint contains no CiCo core parameters")
        return core
    return state


def build_retriever(
    config: dict[str, Any],
    *,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[PMGRRetriever, Any]:
    checkpoint_path = Path(checkpoint or config["paths"]["initialization_checkpoint"])
    if not checkpoint_path.is_file():
        raise CheckpointContractError(f"checkpoint is unavailable: {checkpoint_path}")
    if checkpoint_path.resolve() == Path(config["paths"]["initialization_checkpoint"]).resolve():
        expected_hash = config["paths"]["initialization_checkpoint_sha256"]
        if sha256_file(checkpoint_path) != expected_hash:
            raise CheckpointContractError("initialization checkpoint SHA-256 mismatch")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    baseline = _baseline_config(config)
    cico_root = Path(config["paths"]["cico_root"]).resolve()
    sys.path.insert(0, str(cico_root))
    try:
        core = _load_cico_core_from_state(
            baseline, _core_state(raw), cico_root=cico_root, device=device
        )
    finally:
        try:
            sys.path.remove(str(cico_root))
        except ValueError:
            pass
    # The historical constructor converts many CLIP tensors to half precision. PMGR's
    # reference/replay engine is explicitly FP32; this also avoids unsupported CPU half backward.
    model = PMGRRetriever(core).to(device).float()
    if isinstance(raw, dict) and raw.get("format") in {
        "pmgr-training-v1",
        "pmgr-training-v2",
    }:
        incompatible = model.load_state_dict(_state_dict(raw), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise CheckpointContractError(f"PMGR checkpoint mismatch: {incompatible}")
    return model, raw


def load_tokenizer(config: dict[str, Any]):
    baseline = _baseline_config(config)
    return load_cico_tokenizer_components(baseline)[0]
