from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dive.data.text_units import MappedTextUnit
from dive.models.evidence import state_hash

from .base import NativeTextFeatures, NativeVideoFeatures, PrelogitScores
from .seds_reproduction import (
    PINNED_SEDS_COMMIT,
    SedsReproduction,
    SedsReproductionError,
    load_seds_reproduction,
    verify_seds_checkout,
)


class SedsAdapterError(ValueError):
    """The pinned SEDS model/batch does not satisfy DIVE's adapter contract."""


@dataclass(frozen=True)
class SedsVideoBatch:
    sample_ids: tuple[str, ...]
    right_pose: Tensor
    left_pose: Tensor
    body_pose: Tensor
    clip_starts: Tensor
    legacy_video_mask: Tensor
    rgb_features: Tensor
    grid_id: str
    raw_frame_counts: tuple[int, ...]
    frames_per_second: tuple[float, ...] | None = None
    pose_raw_frame_indices: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class SedsTextBatch:
    text_ids: tuple[str, ...]
    input_ids: Tensor
    token_type_ids: Tensor
    attention_mask: Tensor


def _unique_nonempty(ids: Sequence[str], name: str) -> tuple[str, ...]:
    values = tuple(str(item) for item in ids)
    if not values or any(not item for item in values) or len(values) != len(set(values)):
        raise SedsAdapterError(f"{name} must be unique nonempty IDs")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> Mapping[str, Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SedsAdapterError("SEDS checkpoint cannot be loaded safely as weights") from exc
    if not isinstance(payload, Mapping):
        raise SedsAdapterError("SEDS checkpoint payload is not a state mapping")
    candidate = payload.get("state_dict", payload)
    if not isinstance(candidate, Mapping) or not candidate:
        raise SedsAdapterError("SEDS checkpoint has no model state")
    if not all(isinstance(name, str) and isinstance(value, Tensor) for name, value in candidate.items()):
        raise SedsAdapterError("SEDS checkpoint model state must map names to tensors")
    state = dict(candidate)
    if all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    return state


def _assert_checkpoint_matches_model(path: Path, model: nn.Module) -> None:
    checkpoint_state = _checkpoint_state(path)
    model_state = model.state_dict()
    if checkpoint_state.keys() != model_state.keys():
        missing = sorted(model_state.keys() - checkpoint_state.keys())[:5]
        unexpected = sorted(checkpoint_state.keys() - model_state.keys())[:5]
        raise SedsAdapterError(
            f"constructed SEDS model does not match checkpoint keys; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for name, model_value in model_state.items():
        checkpoint_value = checkpoint_state[name]
        if model_value.shape != checkpoint_value.shape or not torch.equal(
            model_value.detach().cpu(), checkpoint_value.detach().cpu()
        ):
            raise SedsAdapterError(f"constructed SEDS model differs from checkpoint at {name}")


def _load_reproduction_config(path: Path, upstream_root: Path) -> SedsReproduction:
    try:
        return load_seds_reproduction(path, upstream_root=upstream_root)
    except SedsReproductionError as exc:
        raise SedsAdapterError(str(exc)) from exc


def _verify_checkout(upstream_root: Path) -> None:
    try:
        verify_seds_checkout(upstream_root)
    except SedsReproductionError as exc:
        raise SedsAdapterError(str(exc)) from exc


def _official_task_config(
    reproduction: SedsReproduction,
    baseline: Mapping[str, Any],
    root: Path,
    *,
    init_sign_model: Path | None,
) -> SimpleNamespace:
    model_arguments = dict(reproduction.model_arguments)
    model_arguments.update(
        {
            "cache_dir": str(root / ".cache"),
            "distributed": False,
            "dual_mix": float(baseline.get("dual_mix", -1)),
            "init_sign_model": None if init_sign_model is None else str(init_sign_model),
            "local_rank": 0,
        }
    )
    return SimpleNamespace(**model_arguments)


def _construct_official_model(
    root: Path,
    task_config: SimpleNamespace,
    *,
    state: Mapping[str, Tensor] | None,
    device: torch.device,
) -> nn.Module:
    existing_modules = sys.modules.get("modules")
    if existing_modules is not None:
        module_file = Path(getattr(existing_modules, "__file__", "")).resolve()
        if root not in module_file.parents:
            raise SedsAdapterError("a conflicting top-level 'modules' package is already imported")
    sys.path.insert(0, str(root))
    previous_bytecode_policy = sys.dont_write_bytecode
    previous_cuda_device = torch.cuda.current_device()
    sys.dont_write_bytecode = True
    try:
        modeling = importlib.import_module("modules.modeling")
        torch.cuda.set_device(device)
        model = modeling.CLIP4Clip.from_pretrained(
            task_config.cross_model,
            cache_dir=task_config.cache_dir,
            distributed=False,
            state_dict=state,
            task_config=task_config,
        )
    except Exception as exc:
        raise SedsAdapterError("official pinned SEDS model construction failed") from exc
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
        torch.cuda.set_device(previous_cuda_device)
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    # The correctness profile is explicitly FP32; upstream convert_weights() casts many CLIP
    # tensors to fp16 even without AMP, so undo that implicit precision change here.
    return model.to(device).float()


def _validate_baseline_identity(
    baseline: Mapping[str, Any], reproduction: SedsReproduction, data: Mapping[str, Any]
) -> None:
    if (
        baseline.get("family") != "seds"
        or baseline.get("upstream_commit") != PINNED_SEDS_COMMIT
        or baseline.get("score_branch") != "fusion"
        or baseline.get("score_scale") != "prelogit"
        or float(baseline.get("dual_mix", -1)) != 0.5
        or data.get("preparation_protocol") != reproduction.controlled_protocol["name"]
    ):
        raise SedsAdapterError("resolved config differs from the controlled SEDS identity")


def _assert_binary_mask(mask: Tensor, shape: tuple[int, int], name: str) -> None:
    if mask.shape != shape or mask.ndim != 2:
        raise SedsAdapterError(f"{name} must have shape {shape}")
    if mask.dtype == torch.bool:
        return
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise SedsAdapterError(f"{name} must contain only binary values")


def normalize_seds_video_mask(legacy_mask: Tensor, *, local_length: int | None = None) -> Tensor:
    """Convert SEDS 0=valid video masks to DIVE bool True=valid masks.

    If `local_length` is provided, remove exactly the one leading CLS position after
    asserting that the remaining positions match the local RGB/pose sequence.
    """
    if legacy_mask.ndim != 2:
        raise SedsAdapterError("SEDS video mask must be rank two")
    _assert_binary_mask(legacy_mask, tuple(legacy_mask.shape), "SEDS video mask")
    if not bool((legacy_mask[:, 0] == 0).all()):
        raise SedsAdapterError("SEDS video mask must begin with one valid CLS position")
    if local_length is None:
        return legacy_mask == 0
    if local_length <= 0 or legacy_mask.shape[1] != local_length + 1:
        raise SedsAdapterError("SEDS mask must contain exactly one CLS plus local positions")
    return legacy_mask[:, 1:] == 0


def normalize_seds_text_mask(legacy_mask: Tensor, token_shape: tuple[int, int]) -> Tensor:
    _assert_binary_mask(legacy_mask, token_shape, "SEDS text mask")
    validity = legacy_mask == 1
    if not bool(validity.any(dim=1).all()):
        raise SedsAdapterError("every SEDS text must contain at least one valid token")
    return validity


def seds_local_rgb_layout(rgb: Tensor) -> Tensor:
    if rgb.ndim == 4:
        if rgb.shape[-1] != 1:
            raise SedsAdapterError("SEDS RGB cache must end in a singleton spatial dimension")
        return rgb.squeeze(-1).transpose(1, 2).contiguous()
    if rgb.ndim == 3:
        return rgb.contiguous()
    raise SedsAdapterError("SEDS RGB cache must have shape [B,D,N,1] or [B,N,D]")


def _safe_normalize(features: Tensor, validity: Tensor, name: str) -> Tensor:
    if features.ndim != 3 or validity.shape != features.shape[:2] or validity.dtype != torch.bool:
        raise SedsAdapterError(f"{name} features/mask shapes disagree")
    valid = features[validity]
    if not torch.isfinite(valid).all():
        raise SedsAdapterError(f"{name} contains NaN/Inf at valid positions")
    if valid.numel() and bool((torch.linalg.vector_norm(valid, dim=-1) <= 1e-12).any()):
        raise SedsAdapterError(f"{name} contains a collapsed valid feature")
    normalized = F.normalize(features.float(), dim=-1)
    return torch.where(validity[..., None], normalized, torch.zeros_like(normalized))


def _masked_expected_scores(
    video: Tensor,
    text: Tensor,
    video_validity: Tensor,
    text_validity: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    if temperature <= 0:
        raise SedsAdapterError("SEDS alignment temperature must be positive")
    video = _safe_normalize(video, video_validity, "SEDS fusion video")
    text = _safe_normalize(text, text_validity, "SEDS native text")
    similarities = torch.einsum("vnd,tmd->vtnm", video, text)
    text_mask = text_validity[None, :, None, :]
    video_mask = video_validity[:, None, :, None]
    # Correct the upstream padding leak by masking before each directional softmax.
    video_to_text_alignment = torch.softmax(
        (similarities / temperature).masked_fill(~text_mask, float("-inf")), dim=-1
    )
    per_video_token = (video_to_text_alignment * similarities).sum(dim=-1)
    i2t = (per_video_token * video_validity[:, None, :]).sum(dim=-1) / video_validity.sum(
        dim=-1
    )[:, None]
    text_to_video_alignment = torch.softmax(
        (similarities / temperature).masked_fill(~video_mask, float("-inf")), dim=-2
    )
    per_text_token = (text_to_video_alignment * similarities).sum(dim=-2)
    t2i = (per_text_token * text_validity[None, :, :]).sum(dim=-1) / text_validity.sum(
        dim=-1
    )[None, :]
    return i2t, t2i


def seds_prelogit_fusion_scores(
    fusion_hidden: Tensor,
    text_hidden: Tensor,
    video_validity: Tensor,
    text_validity: Tensor,
    *,
    dual_mix: float = 0.5,
    temperature: float = 0.07,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return mixed prelogit S0 plus both unscaled directional matrices."""
    if not 0 <= dual_mix <= 1:
        raise SedsAdapterError("SEDS dual_mix must be in [0,1]")
    i2t, t2i = _masked_expected_scores(
        fusion_hidden,
        text_hidden,
        video_validity,
        text_validity,
        temperature=temperature,
    )
    # Both upstream directional matrices have [video,text] orientation.
    mixed = dual_mix * i2t + (1.0 - dual_mix) * t2i
    if not torch.isfinite(mixed).all() or bool((mixed.abs() > 1.0 + 1e-5).any()):
        raise SedsAdapterError("SEDS prelogit score escaped its cosine bounds")
    return mixed, i2t, t2i


def seds_prelogit_paired_scores(
    fusion_hidden: Tensor,
    text_hidden: Tensor,
    video_validity: Tensor,
    text_validity: Tensor,
    *,
    dual_mix: float = 0.5,
    temperature: float = 0.07,
) -> tuple[Tensor, Tensor, Tensor]:
    """Score aligned video/text pairs without materializing a Cartesian pair matrix."""
    if fusion_hidden.ndim != 3 or text_hidden.ndim != 3:
        raise SedsAdapterError("paired SEDS inputs must have shapes [P,N,D]")
    if fusion_hidden.shape[0] != text_hidden.shape[0]:
        raise SedsAdapterError("paired SEDS inputs must have the same pair count")
    if fusion_hidden.shape[-1] != text_hidden.shape[-1]:
        raise SedsAdapterError("paired SEDS feature dimensions differ")
    if video_validity.shape != fusion_hidden.shape[:2] or video_validity.dtype != torch.bool:
        raise SedsAdapterError("paired SEDS video mask is invalid")
    if text_validity.shape != text_hidden.shape[:2] or text_validity.dtype != torch.bool:
        raise SedsAdapterError("paired SEDS text mask is invalid")
    if not 0 <= dual_mix <= 1 or temperature <= 0:
        raise SedsAdapterError("paired SEDS mixing/temperature is invalid")
    if not bool(video_validity.any(dim=1).all()) or not bool(text_validity.any(dim=1).all()):
        raise SedsAdapterError("paired SEDS inputs contain an empty valid sequence")
    video = _safe_normalize(fusion_hidden, video_validity, "paired SEDS fusion video")
    text = _safe_normalize(text_hidden, text_validity, "paired SEDS native text")
    similarities = torch.einsum("pnd,pmd->pnm", video, text)
    text_mask = text_validity[:, None, :]
    video_mask = video_validity[:, :, None]
    video_to_text = torch.softmax(
        (similarities / temperature).masked_fill(~text_mask, float("-inf")), dim=-1
    )
    i2t = ((video_to_text * similarities).sum(dim=-1) * video_validity).sum(dim=-1)
    i2t = i2t / video_validity.sum(dim=-1)
    text_to_video = torch.softmax(
        (similarities / temperature).masked_fill(~video_mask, float("-inf")), dim=-2
    )
    t2i = ((text_to_video * similarities).sum(dim=-2) * text_validity).sum(dim=-1)
    t2i = t2i / text_validity.sum(dim=-1)
    mixed = dual_mix * i2t + (1.0 - dual_mix) * t2i
    if not torch.isfinite(mixed).all() or bool((mixed.abs() > 1.0 + 1e-5).any()):
        raise SedsAdapterError("paired SEDS prelogit score escaped its cosine bounds")
    return mixed, i2t, t2i


class SedsLocalPoseEncoder(nn.Module):
    """Clone of the SEDS GCN/sign-conv path before the global CLIP transformer."""

    def __init__(self, signbert: nn.Module, *, slide_windows: int = 16) -> None:
        super().__init__()
        if slide_windows <= 0:
            raise SedsAdapterError("SEDS slide window must be positive")
        self.signbert = signbert
        self.slide_windows = int(slide_windows)

    def forward(self, pose: Mapping[str, Tensor], grid: Tensor) -> Tensor:
        if set(pose) != {"right", "left", "body"}:
            raise SedsAdapterError("SEDS local pose requires right/left/body streams")
        batch_sizes = {value.shape[0] for value in pose.values()}
        frame_lengths = {value.shape[1] for value in pose.values()}
        if len(batch_sizes) != 1 or len(frame_lengths) != 1:
            raise SedsAdapterError("SEDS pose streams must be temporally synchronized")
        batch_size = next(iter(batch_sizes))
        frame_length = next(iter(frame_lengths))
        if frame_length < self.slide_windows:
            raise SedsAdapterError("SEDS pose streams must be padded to at least one full window")
        if grid.ndim != 3 or grid.shape[0] != batch_size or grid.shape[-1] != 2:
            raise SedsAdapterError("SEDS local grid must have shape [B,N,2]")
        embedded = self.signbert.gcn_emb(dict(pose))
        if not isinstance(embedded, Mapping) or "feat" not in embedded:
            raise SedsAdapterError("SEDS gcn_emb did not return the expected feat stream")
        frame_features = embedded["feat"]
        if frame_features.shape[:2] != (batch_size, frame_length):
            raise SedsAdapterError("SEDS GCN output changed time or batch layout")
        samples: list[Tensor] = []
        for batch_index in range(batch_size):
            windows: list[Tensor] = []
            for left_tensor, right_tensor in grid[batch_index]:
                left, right = int(left_tensor), int(right_tensor)
                if left < 0 and right < 0:
                    windows.append(torch.zeros_like(frame_features[batch_index, : self.slide_windows]))
                    continue
                if left < 0 or right > frame_length or right - left != self.slide_windows:
                    raise SedsAdapterError(
                        "SEDS canonical pose windows must be valid fixed-length half-open intervals"
                    )
                windows.append(frame_features[batch_index, left:right])
            samples.append(torch.stack(windows))
        window_features = torch.stack(samples)
        batch, clips, window, dimension = window_features.shape
        convolved = self.signbert.sign_conv(window_features.reshape(batch * clips, window, dimension))
        if convolved.shape != (batch * clips, window, dimension):
            raise SedsAdapterError("SEDS sign_conv changed the expected local feature layout")
        return convolved.reshape(batch, clips, window, dimension).mean(dim=2)


class SedsAdapter:
    """Side-effect-free wrapper around an already constructed pinned SEDS model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        upstream_root: str | Path,
        dual_mix: float = 0.5,
        temperature: float = 0.07,
    ) -> None:
        self.model = model
        self.upstream_root = Path(upstream_root).resolve()
        self.dual_mix = float(dual_mix)
        self.temperature = float(temperature)
        if not 0 <= self.dual_mix <= 1 or self.temperature <= 0:
            raise SedsAdapterError("invalid SEDS score configuration")
        for attribute in ("get_visual_output", "get_sequence_output", "fusion", "signbert", "clip"):
            if not hasattr(model, attribute):
                raise SedsAdapterError(f"SEDS model is missing required attribute: {attribute}")
        self._checkpoint_metadata: dict[str, Any] | None = None

    @property
    def checkpoint_metadata(self) -> Mapping[str, Any]:
        if self._checkpoint_metadata is None:
            raise SedsAdapterError("SEDS checkpoint has not been validated")
        return dict(self._checkpoint_metadata)

    @classmethod
    def from_official_checkpoint(
        cls,
        checkpoint_path: str | Path,
        resolved_config: Mapping[str, Any],
        *,
        upstream_root: str | Path,
        device: str | torch.device = "cuda",
        temperature: float = 0.07,
    ) -> "SedsAdapter":
        """Construct the official pinned model without importing its training entry point."""
        root = Path(upstream_root).resolve()
        _verify_checkout(root)
        baseline = resolved_config.get("baseline", {})
        reproduction_path = baseline.get("reproduction_config")
        if not reproduction_path or not Path(reproduction_path).is_file():
            raise SedsAdapterError("controlled SEDS reproduction config is missing")
        reproduction = _load_reproduction_config(Path(reproduction_path), root)
        data = resolved_config.get("data", {})
        if not isinstance(data, Mapping):
            raise SedsAdapterError("resolved data config must be a mapping")
        _validate_baseline_identity(baseline, reproduction, data)
        clip_weights = root / reproduction.external_assets["clip_initialization"]
        if not clip_weights.is_file():
            raise SedsAdapterError(f"official SEDS CLIP initialization is missing: {clip_weights}")
        target_device = torch.device(device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise SedsAdapterError(
                "pinned SEDS construction requires CUDA because upstream graph buffers call .cuda()"
            )
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise SedsAdapterError(f"SEDS checkpoint is missing: {checkpoint}")
        state = dict(_checkpoint_state(checkpoint))
        if not any(name.startswith("signbert.") for name in state):
            raise SedsAdapterError("locked SEDS checkpoint does not contain the SignBERT branch")
        task_config = _official_task_config(
            reproduction, baseline, root, init_sign_model=None
        )
        model = _construct_official_model(
            root, task_config, state=state, device=target_device
        )
        adapter = cls(
            model,
            upstream_root=root,
            dual_mix=float(baseline.get("dual_mix", -1)),
            temperature=temperature,
        )
        adapter.load_and_validate(checkpoint, resolved_config)
        return adapter

    @staticmethod
    def build_official_training_model(
        resolved_config: Mapping[str, Any],
        *,
        upstream_root: str | Path,
        device: str | torch.device = "cuda",
    ) -> tuple[nn.Module, Mapping[str, Any]]:
        """Construct trainable controlled B0 from the separately released native initial assets."""
        root = Path(upstream_root).resolve()
        _verify_checkout(root)
        baseline = resolved_config.get("baseline", {})
        data = resolved_config.get("data", {})
        if not isinstance(baseline, Mapping) or not isinstance(data, Mapping):
            raise SedsAdapterError("resolved baseline/data configs must be mappings")
        reproduction_path = baseline.get("reproduction_config")
        if not reproduction_path or not Path(reproduction_path).is_file():
            raise SedsAdapterError("controlled SEDS reproduction config is missing")
        reproduction_file = Path(reproduction_path)
        reproduction = _load_reproduction_config(reproduction_file, root)
        _validate_baseline_identity(baseline, reproduction, data)
        clip_weights = root / reproduction.external_assets["clip_initialization"]
        configured_initial = baseline.get("initial_weights")
        if not configured_initial:
            raise SedsAdapterError("baseline.initial_weights is not configured")
        signbert_weights = Path(str(configured_initial)).resolve()
        expected_signbert = (root / reproduction.external_assets["signbert_initialization"]).resolve()
        if signbert_weights != expected_signbert:
            raise SedsAdapterError("SignBERT initialization path differs from reproduction contract")
        for name, path in (
            ("CLIP initialization", clip_weights),
            ("SignBERT initialization", signbert_weights),
        ):
            if not path.is_file():
                raise SedsAdapterError(f"official SEDS {name} is missing: {path}")
        try:
            initialization = torch.load(signbert_weights, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SedsAdapterError("SignBERT initialization cannot be loaded safely") from exc
        if not isinstance(initialization, Mapping) or not isinstance(
            initialization.get("state_dict"), Mapping
        ):
            raise SedsAdapterError("SignBERT initialization has no state_dict mapping")
        target_device = torch.device(device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise SedsAdapterError(
                "pinned SEDS construction requires CUDA because upstream graph buffers call .cuda()"
            )
        task_config = _official_task_config(
            reproduction,
            baseline,
            root,
            init_sign_model=None,
        )
        model = _construct_official_model(root, task_config, state=None, device=target_device)
        raw_signbert_state = initialization["state_dict"]
        signbert_state = (
            raw_signbert_state["GCN_Transform"]
            if isinstance(raw_signbert_state.get("GCN_Transform"), Mapping)
            else raw_signbert_state
        )
        try:
            incompatible = model.signbert.load_state_dict(signbert_state, strict=False)
        except Exception as exc:
            raise SedsAdapterError("SignBERT initialization tensors are incompatible") from exc
        if incompatible.unexpected_keys:
            raise SedsAdapterError(
                f"SignBERT initialization has unexpected keys: {incompatible.unexpected_keys[:5]}"
            )
        model.train().requires_grad_(True)
        metadata = {
            "schema_version": "seds_training_initialization.v1",
            "upstream_commit": PINNED_SEDS_COMMIT,
            "reproduction_config_path": str(reproduction_file.resolve()),
            "reproduction_config_sha256": _sha256(reproduction_file),
            "clip_initialization_path": str(clip_weights.resolve()),
            "clip_initialization_sha256": _sha256(clip_weights),
            "signbert_initialization_path": str(signbert_weights),
            "signbert_initialization_sha256": _sha256(signbert_weights),
            "initial_model_state_sha256": state_hash(model),
            "signbert_missing_keys": sorted(incompatible.missing_keys),
            "signbert_unexpected_keys": [],
            "device": str(target_device),
        }
        return model, metadata

    def _verify_upstream(self) -> None:
        _verify_checkout(self.upstream_root)

    def load_and_validate(
        self, checkpoint_path: Path, resolved_config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Validate provenance and freeze a model whose checkpoint was loaded by its factory.

        SEDS construction needs the separately distributed CLIP/SignBERT assets. The
        caller/factory constructs the official model first; this method verifies the locked
        checkpoint/config contract rather than silently synthesizing missing upstream assets.
        """
        self._verify_upstream()
        baseline = resolved_config.get("baseline", {})
        if (
            baseline.get("family") != "seds"
            or baseline.get("upstream_commit") != PINNED_SEDS_COMMIT
            or baseline.get("score_branch") != "fusion"
            or baseline.get("score_scale") != "prelogit"
            or float(baseline.get("dual_mix", -1)) != self.dual_mix
        ):
            raise SedsAdapterError("resolved baseline config does not match the pinned SEDS adapter")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise SedsAdapterError(f"SEDS checkpoint is missing: {checkpoint}")
        configured_checkpoint = baseline.get("locked_checkpoint")
        if not configured_checkpoint or Path(configured_checkpoint).resolve() != checkpoint.resolve():
            raise SedsAdapterError("loaded SEDS checkpoint differs from baseline.locked_checkpoint")
        reproduction_path = baseline.get("reproduction_config")
        if not reproduction_path or not Path(reproduction_path).is_file():
            raise SedsAdapterError("controlled SEDS reproduction config is missing")
        reproduction_file = Path(reproduction_path)
        reproduction = _load_reproduction_config(reproduction_file, self.upstream_root)
        model_arguments = reproduction.model_arguments
        data = resolved_config.get("data", {})
        if not isinstance(data, Mapping) or data.get("preparation_protocol") != (
            reproduction.controlled_protocol["name"]
        ):
            raise SedsAdapterError("DIVE data protocol differs from the SEDS reproduction contract")
        task_config = getattr(self.model, "task_config", None)
        model_contract = {
            "slide_windows": getattr(task_config, "slide_windows", None),
            "feature_len": getattr(task_config, "feature_len", None),
            "fusion_type": getattr(self.model, "fusion_type", None),
            "sim_header": getattr(self.model, "sim_header", None),
            "dual_mix": getattr(self.model, "dual_mix", None),
        }
        expected_model_contract = {
            "slide_windows": model_arguments["slide_windows"],
            "feature_len": model_arguments["feature_len"],
            "fusion_type": model_arguments["fusion_type"],
            "sim_header": model_arguments["sim_header"],
            "dual_mix": self.dual_mix,
        }
        if task_config is None or model_contract != expected_model_contract:
            raise SedsAdapterError("constructed SEDS model task config differs from reproduction")
        _assert_checkpoint_matches_model(checkpoint, self.model)
        self.model.eval().requires_grad_(False)
        self._checkpoint_metadata = {
            "schema_version": "seds_adapter.v1",
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "model_state_sha256": state_hash(self.model),
            "upstream_commit": PINNED_SEDS_COMMIT,
            "reproduction_config_path": str(reproduction_file.resolve()),
            "reproduction_config_sha256": _sha256(reproduction_file),
            "score_branch": "fusion",
            "score_scale": "prelogit",
            "dual_mix": self.dual_mix,
            "alignment_temperature": self.temperature,
        }
        return dict(self._checkpoint_metadata)

    @staticmethod
    def _video_inputs(batch: SedsVideoBatch) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        right = {"pose": batch.right_pose}
        left = {"pose": batch.left_pose}
        body = {
            "pose": batch.body_pose,
            "clips_start": batch.clip_starts,
            "mask": batch.legacy_video_mask,
            "rgb": batch.rgb_features,
        }
        return right, left, body

    def encode_video_native(self, video_batch: SedsVideoBatch) -> NativeVideoFeatures:
        ids = _unique_nonempty(video_batch.sample_ids, "SEDS video sample_ids")
        if len(ids) != video_batch.legacy_video_mask.shape[0]:
            raise SedsAdapterError("SEDS video IDs do not match batch size")
        self.model.eval()
        with torch.no_grad():
            legacy_mask, pose_hidden, rgb_hidden = self.model.get_visual_output(
                *self._video_inputs(video_batch), shaped=True, get_hidden=True
            )
            if not torch.equal(legacy_mask, video_batch.legacy_video_mask):
                raise SedsAdapterError("SEDS visual encoder returned a different legacy mask")
            validity = normalize_seds_video_mask(legacy_mask)
            if pose_hidden.shape != rgb_hidden.shape or pose_hidden.shape[:2] != validity.shape:
                raise SedsAdapterError("SEDS contextual pose/RGB outputs or mask disagree")
            fusion = self.model.fusion(pose_hidden, rgb_hidden, legacy_mask)
            normalized = _safe_normalize(fusion, validity, "SEDS fusion video")
            pooled = (normalized * validity[..., None]).sum(dim=1) / validity.sum(dim=1)[:, None]
            pooled = F.normalize(pooled, dim=-1)
        return NativeVideoFeatures(
            sample_ids=ids,
            pooled=pooled,
            validity=validity,
            streams={
                "pose_hidden": pose_hidden,
                "rgb_hidden": rgb_hidden,
                "fusion_hidden": fusion,
            },
            metadata={"grid_id": video_batch.grid_id, "legacy_mask_convention": "0_valid"},
        )

    def encode_text_native(self, text_batch: SedsTextBatch) -> NativeTextFeatures:
        ids = _unique_nonempty(text_batch.text_ids, "SEDS text_ids")
        if len(ids) != text_batch.input_ids.shape[0]:
            raise SedsAdapterError("SEDS text IDs do not match batch size")
        self.model.eval()
        with torch.no_grad():
            legacy_mask, hidden = self.model.get_sequence_output(
                text_batch.input_ids,
                text_batch.token_type_ids,
                text_batch.attention_mask,
                shaped=True,
                get_hidden=True,
            )
            validity = normalize_seds_text_mask(legacy_mask, tuple(hidden.shape[:2]))
            normalized = _safe_normalize(hidden, validity, "SEDS native text")
            pooled = (normalized * validity[..., None]).sum(dim=1) / validity.sum(dim=1)[:, None]
            pooled = F.normalize(pooled, dim=-1)
        return NativeTextFeatures(
            text_ids=ids,
            pooled=pooled,
            token_features=hidden,
            token_validity=validity,
            metadata={"legacy_mask_convention": "1_valid"},
        )

    def score_prelogit(
        self, video_features: NativeVideoFeatures, text_features: NativeTextFeatures
    ) -> PrelogitScores:
        required_streams = {"pose_hidden", "rgb_hidden"}
        if not required_streams <= set(video_features.streams):
            raise SedsAdapterError("SEDS video features lack exact pose/RGB contextual streams")
        self.model.eval()
        with torch.no_grad():
            # Fusion remains the locked native selected branch; scoring itself is pure and
            # masks padding before softmax rather than reproducing the upstream padding leak.
            fusion = video_features.streams.get("fusion_hidden")
            if fusion is None:
                legacy_mask = (~video_features.validity).to(torch.long)
                fusion = self.model.fusion(
                    video_features.streams["pose_hidden"],
                    video_features.streams["rgb_hidden"],
                    legacy_mask,
                )
            mixed, i2t, t2i = seds_prelogit_fusion_scores(
                fusion,
                text_features.token_features,
                video_features.validity,
                text_features.token_validity,
                dual_mix=self.dual_mix,
                temperature=self.temperature,
            )
            logit_scale = self.model.clip.logit_scale.exp().detach().float()
        quantiles = torch.quantile(mixed.float().reshape(-1), torch.tensor([0.0, 0.5, 1.0]))
        return PrelogitScores(
            video_ids=video_features.sample_ids,
            text_ids=text_features.text_ids,
            scores=mixed,
            logit_scale=logit_scale,
            directional_scores={"i2t": i2t, "t2i": t2i},
            diagnostics={
                "minimum": float(quantiles[0]),
                "median": float(quantiles[1]),
                "maximum": float(quantiles[2]),
                "logit_scale": float(logit_scale),
                "score_orientation": "video_rows_text_columns",
            },
        )

    def validate_unpadded_native_parity(
        self,
        video_features: NativeVideoFeatures,
        text_features: NativeTextFeatures,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> Mapping[str, Any]:
        """Compare DIVE to the released scorer after removing every padded position."""
        if not hasattr(self.model, "get_similarity_logits"):
            raise SedsAdapterError("SEDS model lacks the native similarity dispatcher")
        if not {"pose_hidden", "rgb_hidden"} <= set(video_features.streams):
            raise SedsAdapterError("SEDS parity probe requires exact pose/RGB contextual streams")
        video_counts = video_features.validity.sum(dim=1)
        text_counts = text_features.token_validity.sum(dim=1)
        video_prefix = (
            torch.arange(video_features.validity.shape[1], device=video_counts.device)[None]
            < video_counts[:, None]
        )
        text_prefix = (
            torch.arange(text_features.token_validity.shape[1], device=text_counts.device)[None]
            < text_counts[:, None]
        )
        if not torch.equal(video_features.validity, video_prefix) or not torch.equal(
            text_features.token_validity, text_prefix
        ):
            raise SedsAdapterError("SEDS valid positions must form contiguous prefixes")
        video_length = int(video_counts.min())
        text_length = int(text_counts.min())
        if video_length <= 0 or text_length <= 0:
            raise SedsAdapterError("SEDS parity probe has no common unpadded prefix")
        video_mask = video_features.validity[:, :video_length]
        text_mask = text_features.token_validity[:, :text_length]
        if not bool(video_mask.all()) or not bool(text_mask.all()):
            raise SedsAdapterError("SEDS parity trim retained a padded position")
        pose = video_features.streams["pose_hidden"][:, :video_length]
        rgb = video_features.streams["rgb_hidden"][:, :video_length]
        text = text_features.token_features[:, :text_length]
        legacy_video_mask = torch.zeros_like(video_mask, dtype=torch.long)
        legacy_text_mask = torch.ones_like(text_mask, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            native = self.model.get_similarity_logits(
                text,
                pose,
                rgb,
                legacy_text_mask,
                legacy_video_mask,
                shaped=True,
                loose_type=getattr(self.model, "loose_type", False),
                is_train=True,
            )
            if not isinstance(native, tuple) or len(native) < 2:
                raise SedsAdapterError("native SEDS scorer returned an invalid parity payload")
            native_i2t, native_t2i = native[:2]
            if not isinstance(native_i2t, Tensor) or not isinstance(native_t2i, Tensor):
                raise SedsAdapterError("native SEDS scorer did not return directional tensors")
            fusion = self.model.fusion(pose, rgb, legacy_video_mask)
            _, dive_i2t, dive_t2i = seds_prelogit_fusion_scores(
                fusion,
                text,
                video_mask,
                text_mask,
                dual_mix=self.dual_mix,
                temperature=self.temperature,
            )
            scale = self.model.clip.logit_scale.exp().detach().float()
            native_i2t_prelogit = native_i2t.float() / scale
            native_t2i_prelogit = native_t2i.float() / scale
        if native_i2t_prelogit.shape != dive_i2t.shape or native_t2i_prelogit.shape != dive_t2i.shape:
            raise SedsAdapterError("native SEDS directional score shapes differ from DIVE")
        i2t_error = float((dive_i2t.float() - native_i2t_prelogit).abs().max())
        t2i_error = float((dive_t2i.float() - native_t2i_prelogit).abs().max())
        passed = torch.allclose(
            dive_i2t.float(), native_i2t_prelogit, atol=atol, rtol=rtol
        ) and torch.allclose(dive_t2i.float(), native_t2i_prelogit, atol=atol, rtol=rtol)
        if not passed:
            raise SedsAdapterError(
                f"native unpadded score parity failed: i2t={i2t_error}, t2i={t2i_error}"
            )
        return {
            "schema_version": "seds_score_parity.v1",
            "passed": True,
            "video_count": len(video_features.sample_ids),
            "text_count": len(text_features.text_ids),
            "unpadded_video_length": video_length,
            "unpadded_text_length": text_length,
            "i2t_max_abs_error": i2t_error,
            "t2i_max_abs_error": t2i_error,
            "atol": atol,
            "rtol": rtol,
            "score_scale": "prelogit",
            "orientation": "video_rows_text_columns",
        }

    def encode_text_units(
        self,
        text_batch: SedsTextBatch,
        unit_mapping: Sequence[Sequence[MappedTextUnit]],
    ) -> NativeTextFeatures:
        native = self.encode_text_native(text_batch)
        return self.pool_text_units(native, unit_mapping)

    def pool_text_units(
        self,
        native: NativeTextFeatures,
        unit_mapping: Sequence[Sequence[MappedTextUnit]],
    ) -> NativeTextFeatures:
        """Pool already-encoded native tokens without rerunning the frozen text encoder."""
        if len(unit_mapping) != len(native.text_ids):
            raise SedsAdapterError("unit mappings do not match the SEDS text batch")
        max_units = max((len(mapping) for mapping in unit_mapping), default=0)
        if max_units == 0:
            raise SedsAdapterError("text-unit encoding requires at least one mapped unit")
        output = native.token_features.new_zeros(
            (len(native.text_ids), max_units, native.token_features.shape[-1])
        )
        validity = torch.zeros(
            (len(native.text_ids), max_units), dtype=torch.bool, device=output.device
        )
        for batch_index, mapping in enumerate(unit_mapping):
            for unit_index, item in enumerate(mapping):
                if not item.complete_after_truncation:
                    continue
                indices = torch.as_tensor(item.subword_indices, device=output.device)
                if indices.numel() == 0 or bool((indices >= native.token_features.shape[1]).any()):
                    raise SedsAdapterError("mapped subword index is empty or out of range")
                if not bool(native.token_validity[batch_index, indices].all()):
                    raise SedsAdapterError("mapped unit references a padded SEDS token")
                pooled = native.token_features[batch_index, indices].float().mean(dim=0)
                if torch.linalg.vector_norm(pooled) <= 1e-12:
                    raise SedsAdapterError("mapped SEDS text unit collapsed before normalization")
                output[batch_index, unit_index] = F.normalize(pooled, dim=-1).to(output.dtype)
                validity[batch_index, unit_index] = True
        return NativeTextFeatures(
            text_ids=native.text_ids,
            pooled=native.pooled,
            token_features=output,
            token_validity=validity,
            metadata={**native.metadata, "unit_pool": "mean_raw_projected_subwords_then_normalize"},
        )

    def rgb_local_features(self, video_batch: SedsVideoBatch, grid_id: str) -> NativeVideoFeatures:
        if grid_id != video_batch.grid_id:
            raise SedsAdapterError("requested RGB grid differs from the prepared video batch")
        ids = _unique_nonempty(video_batch.sample_ids, "SEDS video sample_ids")
        local = seds_local_rgb_layout(video_batch.rgb_features)
        validity = normalize_seds_video_mask(
            video_batch.legacy_video_mask, local_length=local.shape[1]
        )
        normalized = _safe_normalize(local, validity, "SEDS local RGB")
        pooled = (normalized * validity[..., None]).sum(dim=1) / validity.sum(dim=1)[:, None]
        return NativeVideoFeatures(
            sample_ids=ids,
            pooled=F.normalize(pooled, dim=-1),
            validity=validity,
            streams={"rgb_local": local},
            metadata={"grid_id": grid_id, "tap": "get_sign_output.rgb_before_clip_encode_image"},
        )

    def local_pose_grid(self, video_batch: SedsVideoBatch, grid_id: str) -> Tensor:
        """Convert native scalar starts to explicit half-open local pose windows."""
        if grid_id != video_batch.grid_id:
            raise SedsAdapterError("requested pose grid differs from the prepared video batch")
        starts = video_batch.clip_starts
        if starts.ndim != 2:
            raise SedsAdapterError("native SEDS clip starts must have shape [B,N]")
        validity = normalize_seds_video_mask(
            video_batch.legacy_video_mask, local_length=starts.shape[1]
        )
        if not torch.equal(starts >= 0, validity):
            raise SedsAdapterError("native SEDS clip starts disagree with local validity")
        slide_windows = int(getattr(self.model.task_config, "slide_windows", 16))
        ends = torch.where(validity, starts + slide_windows, torch.full_like(starts, -1))
        grid = torch.stack((starts, ends), dim=-1)
        if bool((ends[validity] > video_batch.body_pose.shape[1]).any()):
            raise SedsAdapterError("native SEDS local pose window exceeds padded pose frames")
        return grid

    def clone_local_pose_encoder(self) -> nn.Module:
        slide_windows = int(getattr(self.model.task_config, "slide_windows", 16))
        return SedsLocalPoseEncoder(copy.deepcopy(self.model.signbert), slide_windows=slide_windows)

    def describe_preprocessing(self) -> Mapping[str, Any]:
        return {
            "schema_version": "seds_preprocessing.v1",
            "upstream_commit": PINNED_SEDS_COMMIT,
            "rgb_tap": "get_sign_output.rgb_final_before_clip_rgb.encode_image",
            "pose_tap": "signbert.gcn_emb_then_window_then_sign_conv_mean",
            "pose_raw_mapping": "captured_after_native_subsampling_and_hand_filter",
            "rgb_raw_mapping": "count_aligned_with_native_pose_clips_pending_asset_audit",
            "legacy_video_mask": "0_valid_with_one_leading_cls",
            "legacy_text_mask": "1_valid_through_eot",
            "native_score": "fusion_directional_mix_divided_by_exp_logit_scale",
            "padding_fix": "mask_before_directional_softmax",
        }

    def describe_receptive_field(
        self, video_batch: SedsVideoBatch, grid_id: str
    ) -> Sequence[Mapping[str, Any]]:
        if grid_id != video_batch.grid_id:
            raise SedsAdapterError("requested receptive-field grid differs from batch grid")
        local_length = video_batch.clip_starts.shape[1]
        validity = normalize_seds_video_mask(
            video_batch.legacy_video_mask, local_length=local_length
        )
        frame_count = video_batch.body_pose.shape[1]
        if len(video_batch.raw_frame_counts) != len(video_batch.sample_ids):
            raise SedsAdapterError("raw_frame_counts does not match video batch")
        fps_values = video_batch.frames_per_second
        if fps_values is not None and len(fps_values) != len(video_batch.sample_ids):
            raise SedsAdapterError("frames_per_second does not match video batch")
        raw_index_maps = video_batch.pose_raw_frame_indices
        if raw_index_maps is not None and len(raw_index_maps) != len(video_batch.sample_ids):
            raise SedsAdapterError("pose_raw_frame_indices does not match video batch")
        records: list[dict[str, Any]] = []
        slide_windows = int(getattr(self.model.task_config, "slide_windows", 16))
        # Two upstream temporal GCN blocks give a conservative 9-frame dependency;
        # expand the 16-frame nominal window by four frames on each side.
        for batch_index, sample_id in enumerate(video_batch.sample_ids):
            raw_frame_count = int(video_batch.raw_frame_counts[batch_index])
            if raw_frame_count <= 0:
                raise SedsAdapterError("raw frame count must be positive")
            if raw_index_maps is None:
                if raw_frame_count > frame_count:
                    raise SedsAdapterError(
                        "raw frame count needs a native pose-to-raw index map after preprocessing"
                    )
                raw_indices = tuple(range(raw_frame_count))
                mapping_policy = "identity_pose_video_frames_v1"
            else:
                raw_indices = raw_index_maps[batch_index]
                if (
                    not raw_indices
                    or len(raw_indices) > frame_count
                    or raw_indices[0] < 0
                    or raw_indices[-1] >= raw_frame_count
                    or any(left >= right for left, right in zip(raw_indices, raw_indices[1:]))
                ):
                    raise SedsAdapterError("native pose-to-raw frame index map is invalid")
                mapping_policy = "native_seds_pose_selected_raw_frames_v1"
            for clip_index, start_tensor in enumerate(video_batch.clip_starts[batch_index]):
                if not validity[batch_index, clip_index]:
                    continue
                start = int(start_tensor)
                if start < 0 or start >= len(raw_indices):
                    raise SedsAdapterError("valid SEDS clip start lies outside processed pose frames")
                pose_left_step = max(0, start - 4)
                pose_right_step = min(len(raw_indices), start + slide_windows + 4)
                rgb_left_step = start
                rgb_right_step = min(len(raw_indices), start + slide_windows)
                left = raw_indices[pose_left_step]
                right = raw_indices[pose_right_step - 1] + 1
                rgb_left = raw_indices[rgb_left_step]
                rgb_right = raw_indices[max(rgb_left_step, rgb_right_step - 1)] + 1
                record: dict[str, Any] = {
                    "sample_id": str(sample_id),
                    "clip_index": clip_index,
                    "raw_frame_interval": [left, right],
                    "pose_raw_frame_interval": [left, right],
                    "rgb_raw_frame_interval": [rgb_left, rgb_right],
                    "interval_convention": "half_open",
                    "nominal_pose_frames": slide_windows,
                    "nominal_rgb_frames": slide_windows,
                    "conservative_pose_rf_frames": right - left,
                    "pose_input_step_interval": [pose_left_step, pose_right_step],
                    "raw_mapping_policy": mapping_policy,
                }
                if fps_values is not None:
                    fps = float(fps_values[batch_index])
                    if fps <= 0:
                        raise SedsAdapterError("frames_per_second must be positive")
                    record["raw_seconds_interval"] = [left / fps, right / fps]
                records.append(record)
        return tuple(records)
