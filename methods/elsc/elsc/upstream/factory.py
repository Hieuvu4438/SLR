"""ELSC retriever construction on top of the shared CiCo factory."""

import sys
from pathlib import Path
from typing import Any

import torch

from elsc.models.retriever import ELSCRetriever
from slr_common.upstream.factory import *  # noqa: F403
from slr_common.upstream.factory import (
    CheckpointContractError,
    _load_cico_core_from_state,
    _state_dict,
)
from slr_common.utils import sha256_file


def build_retriever_from_checkpoint(
    config: dict[str, Any], checkpoint: str | Path, *, device: str | torch.device = "cpu"
):
    """Load an upstream CiCo checkpoint or a resumable ELSC checkpoint."""
    checkpoint_path = Path(checkpoint)
    for role in ("init_checkpoint", "teacher_checkpoint"):
        configured = config.get("model", {}).get(role)
        expected = config.get("model", {}).get(f"{role}_sha256")
        if configured and expected and Path(configured).resolve() == checkpoint_path.resolve():
            actual = sha256_file(checkpoint_path)
            if actual != expected:
                raise CheckpointContractError(
                    f"{role} SHA-256 mismatch: expected {expected}, found {actual}"
                )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = _state_dict(raw)
    is_elsc = any(key.startswith("core.") for key in state)
    core_state = (
        {
            key.removeprefix("core."): value
            for key, value in state.items()
            if key.startswith("core.")
        }
        if is_elsc
        else state
    )
    cico_root = Path(config["upstream"]["cico_root"]).resolve()
    sys.path.insert(0, str(cico_root))
    try:
        core = _load_cico_core_from_state(config, core_state, cico_root=cico_root, device=device)
    finally:
        try:
            sys.path.remove(str(cico_root))
        except ValueError:
            pass
    model_cfg = config["model"]
    adapter_cfg = model_cfg["adapter"]
    inferred_text_dim = int(core.clip.text_projection.shape[-1])
    requested_text_dim = model_cfg.get("lexical_head", {}).get(
        "output_dim", "infer_from_checkpoint"
    )
    if requested_text_dim not in {None, "infer", "infer_from_checkpoint", "infer_from_teacher"}:
        if int(requested_text_dim) != inferred_text_dim:
            raise CheckpointContractError(
                "lexical head dimension does not match checkpoint text projection: "
                f"{requested_text_dim} != {inferred_text_dim}"
            )
    retriever = ELSCRetriever(
        core,
        input_dim=int(config["data"]["feature_dim"]),
        hidden_dim=int(adapter_cfg.get("hidden_dim", 256)),
        text_dim=inferred_text_dim,
        core_frozen=bool(model_cfg.get("backbone_frozen", True)),
        adapter_enabled=bool(adapter_cfg.get("enabled", True)),
    ).to(device)
    if is_elsc:
        incompatible = retriever.load_state_dict(state, strict=False)
        allowed = ("local_head.",) if raw.get("format") == "elsc-inference-v1" else ()
        disallowed = [key for key in incompatible.missing_keys if not key.startswith(allowed)]
        if disallowed or incompatible.unexpected_keys:
            raise CheckpointContractError(
                "ELSC checkpoint incompatibility: "
                f"missing={disallowed}, unexpected={incompatible.unexpected_keys}"
            )
    return retriever, raw
