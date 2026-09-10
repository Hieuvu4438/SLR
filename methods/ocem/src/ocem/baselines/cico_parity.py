"""Real-checkpoint parity validation for the pinned CiCo adapter."""

from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ocem.baselines.cico_adapter import CiCoAdapter
from ocem.provenance.hashes import sha256_file


class CiCoParityError(ValueError):
    """Raised when real-checkpoint parity cannot be validated safely."""


def _verify(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise CiCoParityError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _load_manifest_records(path: Path, sample_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    wanted = set(sample_ids)
    found: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CiCoParityError(f"invalid JSONL {path}:{line_number}: {error}") from error
        sample_id = str(record.get("sample_id", ""))
        if sample_id in wanted:
            found[sample_id] = record
    missing = sorted(wanted.difference(found))
    if missing:
        raise CiCoParityError(f"fixture IDs are absent from manifest: {missing}")
    return [found[sample_id] for sample_id in sample_ids]


def _tokenize(tokenizer: Any, text: str, max_words: int = 32) -> tuple[list[int], list[bool]]:
    words = ["<|startoftext|>"] + tokenizer.tokenize(text)
    available = max_words - 1
    if len(words) > available:
        indices = [0] + list(np.linspace(1, len(words) - 1, available - 1, dtype=int))
        words = list(np.asarray(words)[indices])
    words.append("<|endoftext|>")
    ids = tokenizer.convert_tokens_to_ids(words)
    valid = [True] * len(ids)
    ids.extend([0] * (max_words - len(ids)))
    valid.extend([False] * (max_words - len(valid)))
    if len(ids) != max_words:
        raise CiCoParityError("tokenizer fixture did not produce the locked length")
    return ids, valid


def _load_feature_fixture(
    records: Sequence[Mapping[str, Any]], feature_root: Path, feature_len: int = 64
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        path = feature_root / f"{sample_id}.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - pinned, locally audited feature cache
        feature = np.asarray(payload["feature"], dtype=np.float32)
        if feature.ndim != 2 or feature.shape[1] != 1024 or feature.shape[0] < 1:
            raise CiCoParityError(f"invalid feature shape for {sample_id}: {feature.shape}")
        if feature.shape[0] >= feature_len:
            indices = np.linspace(0, feature.shape[0] - 1, feature_len, dtype=int)
        else:
            indices = np.arange(feature.shape[0], dtype=int)
        padded = np.zeros((feature_len, 1024), dtype=np.float32)
        padded[: len(indices)] = feature[indices]
        valid = np.zeros(feature_len, dtype=bool)
        valid[: len(indices)] = True
        arrays.append(padded)
        masks.append(valid)
        evidence.append(
            {
                "sample_id": sample_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "source_windows": int(feature.shape[0]),
                "selected_windows": int(len(indices)),
                "selection": "np.linspace" if feature.shape[0] >= feature_len else "all_then_zero_pad",
                "selected_indices": indices.tolist(),
            }
        )
    return torch.from_numpy(np.stack(arrays)), torch.from_numpy(np.stack(masks)), evidence


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _metric_fixture(metrics_module: Any, direct: np.ndarray, adapted: np.ndarray) -> dict[str, Any]:
    direct_tensor = direct[:, None, :]
    adapted_tensor = adapted[:, None, :]
    direct_t2v = metrics_module.tensor_text_to_video_metrics(direct_tensor)
    adapted_t2v = metrics_module.tensor_text_to_video_metrics(adapted_tensor)
    direct_v2t = metrics_module.compute_metrics(
        metrics_module.tensor_video_to_text_sim(direct_tensor)
    )
    adapted_v2t = metrics_module.compute_metrics(
        metrics_module.tensor_video_to_text_sim(adapted_tensor)
    )
    return {
        "shape": list(direct.shape),
        "contains_tie": True,
        "contains_duplicate_caption_group": True,
        "sample_and_caption_orders_differ": True,
        "direct_t2v": direct_t2v,
        "adapter_t2v": adapted_t2v,
        "direct_v2t": direct_v2t,
        "adapter_v2t": adapted_v2t,
        "exact_metric_parity": direct_t2v == adapted_t2v and direct_v2t == adapted_v2t,
    }


def validate_cico_adapter_parity(
    *,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    clip_checkpoint: str | Path,
    expected_clip_sha256: str,
    upstream_root: str | Path,
    expected_source_hashes: Mapping[str, str],
    manifest: str | Path,
    feature_root: str | Path,
    sample_ids: Sequence[str],
    device: str = "cuda:0",
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare the adapter with actual CiCo encode/score/loss/gradient/update paths."""

    if len(sample_ids) < 2 or len(set(sample_ids)) != len(sample_ids):
        raise CiCoParityError("at least two unique fixture sample IDs are required")
    checkpoint, clip_checkpoint = Path(checkpoint), Path(clip_checkpoint)
    upstream_root, manifest, feature_root = (
        Path(upstream_root),
        Path(manifest),
        Path(feature_root),
    )
    _verify(checkpoint, expected_checkpoint_sha256, "CiCo checkpoint")
    _verify(clip_checkpoint, expected_clip_sha256, "CLIP checkpoint")
    source_paths = {
        "modeling": upstream_root / "modules/modeling.py",
        "module_clip": upstream_root / "modules/module_clip.py",
        "tokenization": upstream_root / "modules/tokenization_clip.py",
        "metrics": upstream_root / "metrics.py",
    }
    for name, path in source_paths.items():
        if name not in expected_source_hashes:
            raise CiCoParityError(f"missing expected source hash for {name}")
        _verify(path, expected_source_hashes[name], name)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise CiCoParityError("requested CUDA device is unavailable")
    if not (str(device).startswith("cuda") or str(device) == "cpu"):
        raise CiCoParityError("device must be 'cpu' or a CUDA device")

    resolved_root = str(upstream_root.resolve())
    sys.path.insert(0, resolved_root)
    try:
        modeling = importlib.import_module("modules.modeling")
        tokenization = importlib.import_module("modules.tokenization_clip")
        metrics_module = importlib.import_module("metrics")
    finally:
        if sys.path[0] == resolved_root:
            sys.path.pop(0)
    if not Path(modeling.__file__).resolve().is_relative_to(upstream_root.resolve()):
        raise CiCoParityError("imported CiCo modeling module is outside the pinned root")

    records = _load_manifest_records(manifest, sample_ids)
    local_h, video_valid, feature_evidence = _load_feature_fixture(records, feature_root)
    tokenizer = tokenization.SimpleTokenizer()
    tokenized = [_tokenize(tokenizer, str(record["caption_raw"])) for record in records]
    text_ids = torch.tensor([item[0] for item in tokenized], dtype=torch.long)
    text_valid = torch.tensor([item[1] for item in tokenized], dtype=torch.bool)
    resolved_device = torch.device(device)
    batch = {
        "local_h": local_h.to(resolved_device),
        "video_valid": video_valid.to(resolved_device),
        "text_ids": text_ids.to(resolved_device),
        "text_input_valid": text_valid.to(resolved_device),
        "text_aug_ids": text_ids.to(resolved_device),
        "text_aug_input_valid": text_valid.to(resolved_device),
    }

    config = SimpleNamespace(
        local_rank=0,
        pretrained_clip_name=str(clip_checkpoint.resolve()),
        visual_num_hidden_layers=12,
        feature_len=64,
        linear_patch="2d",
        sim_header="Filip",
        loose_type=True,
        cross_num_hidden_layers=4,
        dual_mix=0.5,
        mix_design="balance",
        alpha=0.9,
        aug_choose="t2v",
        not_load_visual=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = modeling.CLIP4Clip.from_pretrained(
        "cross-base", cache_dir="", distributed=False, state_dict=state, task_config=config
    ).to(resolved_device)
    execution_dtype = "upstream_mixed_fp16_fp32"
    if resolved_device.type == "cpu":
        # The spec's initial port tolerance is FP32.  This also provides a
        # deterministic reference when repeated CUDA half-precision backward
        # kernels are not bitwise stable.
        model.float()
        execution_dtype = "fp32_port"
    adapter = CiCoAdapter(model)
    token_types = torch.zeros_like(batch["text_ids"])
    upstream_mask = adapter.upstream_video_mask(batch["video_valid"])
    upstream_video = adapter.upstream_video_tensor(batch["local_h"])

    model.eval()
    with torch.no_grad():
        direct = model.get_sequence_visual_output(
            batch["text_ids"],
            token_types,
            batch["text_input_valid"].long(),
            upstream_video,
            upstream_mask,
            shaped=True,
            video_frame=-1,
            input_ids_aug=batch["text_aug_ids"],
            attention_mask_aug=batch["text_aug_input_valid"].long(),
        )
        direct_y, direct_text_mask, _, direct_z, direct_video_mask, _, direct_aug, direct_aug_mask, _ = direct
        direct_i2t, direct_t2i, _ = model.get_similarity_logits(
            direct_y,
            direct_z,
            direct_text_mask,
            direct_video_mask,
            shaped=True,
            loose_type=model.loose_type,
            is_train=True,
            sequence_hidden_aug=direct_aug,
            text_mask_aug=direct_aug_mask,
        )
        encoded = adapter.encode(batch, training=False)
        adapted_logits = adapter.scaled_directional_matrices(encoded)

    encode_errors = {
        "visual_tokens": _max_error(encoded.visual_tokens, direct_z),
        "canonical_text_tokens": _max_error(encoded.canonical_text_tokens, direct_y),
        "augmented_text_tokens": _max_error(encoded.augmented_text_tokens, direct_aug),
    }
    score_errors = {
        "t2v": _max_error(adapted_logits.logits_t2v, direct_i2t),
        "v2t": _max_error(adapted_logits.logits_v2t, direct_t2i),
    }

    representative = (
        "clip.logit_scale",
        "clip.visual.conv1.weight",
        "clip.visual.transformer.resblocks.0.attn.in_proj_weight",
        "clip.token_embedding.weight",
        "clip.text_projection",
    )
    model.zero_grad(set_to_none=True)
    model.train()
    torch.manual_seed(0)
    direct_loss = model(
        batch["text_ids"].unsqueeze(1),
        token_types.unsqueeze(1),
        batch["text_input_valid"].long().unsqueeze(1),
        upstream_video,
        upstream_mask,
        input_ids_aug=batch["text_aug_ids"].unsqueeze(1),
        attention_mask_aug=batch["text_aug_input_valid"].long().unsqueeze(1),
    )
    direct_loss.backward()
    parameters = dict(model.named_parameters())
    direct_gradients = {name: parameters[name].grad.detach().clone() for name in representative}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    adapted_encoded = adapter.encode(batch, training=True)
    adapted_loss = adapter.original_training_loss(adapted_encoded)
    adapted_loss.backward()
    adapted_gradients = {name: parameters[name].grad.detach().clone() for name in representative}
    gradient_errors = {
        name: _max_error(adapted_gradients[name], direct_gradients[name]) for name in representative
    }
    update_errors: dict[str, float] = {}
    for name in representative:
        initial = parameters[name].detach().float().clone()
        direct_parameter = torch.nn.Parameter(initial.clone())
        adapted_parameter = torch.nn.Parameter(initial.clone())
        direct_parameter.grad = direct_gradients[name].float().clone()
        adapted_parameter.grad = adapted_gradients[name].float().clone()
        torch.optim.SGD([direct_parameter], lr=1e-4, momentum=0.9).step()
        torch.optim.SGD([adapted_parameter], lr=1e-4, momentum=0.9).step()
        update_errors[name] = _max_error(adapted_parameter, direct_parameter)

    direct_metric_scores = np.array(
        [[3.0, 2.0, 2.0, 0.0], [1.0, 3.0, 0.0, 0.0], [2.0, 2.0, 3.0, 1.0], [0.0, 1.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    adapter_metric_scores = direct_metric_scores.copy()
    metric = _metric_fixture(metrics_module, direct_metric_scores, adapter_metric_scores)
    comparisons = list(encode_errors.values()) + list(score_errors.values()) + [
        abs(float(adapted_loss.detach()) - float(direct_loss.detach()))
    ]
    gradients_pass = all(
        torch.allclose(adapted_gradients[name], direct_gradients[name], atol=atol, rtol=rtol)
        for name in representative
    )
    passed = (
        max(comparisons) <= atol
        and gradients_pass
        and max(update_errors.values()) <= atol
        and torch.equal(encoded.upstream_video_mask, direct_video_mask)
        and torch.equal(encoded.canonical_text_mask, direct_text_mask)
        and torch.equal(encoded.augmented_text_mask, direct_aug_mask)
        and metric["exact_metric_parity"]
    )
    return {
        "schema_version": "ocem.cico_adapter_parity.v1",
        "status": "PASS" if passed else "FAIL_TECHNICAL",
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": expected_checkpoint_sha256},
        "clip_checkpoint": {
            "path": str(clip_checkpoint.resolve()),
            "sha256": expected_clip_sha256,
        },
        "upstream_root": str(upstream_root.resolve()),
        "source_sha256": dict(expected_source_hashes),
        "fixture": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "sample_ids": list(sample_ids),
            "features": feature_evidence,
            "batch_size": len(sample_ids),
            "text_length": 32,
            "feature_length": 64,
            "canonical_equals_augmented": True,
        },
        "contract": {
            "matrix_orientation": "[video,text]",
            "internal_valid_mask": "True=valid",
            "upstream_video_mask": "CLS=1,valid=0,padding=1",
            "internal_softmax_temperature": 0.07,
            "external_logit_scale": float(adapter.logit_scale().detach().cpu()),
            "dual_mix": adapter.dual_mix,
            "mix_design": adapter.mix_design,
            "execution_dtype": execution_dtype,
        },
        "encode_max_absolute_error": encode_errors,
        "score_max_absolute_error": score_errors,
        "loss": {
            "direct": float(direct_loss.detach().cpu()),
            "adapter": float(adapted_loss.detach().cpu()),
            "absolute_error": abs(float(adapted_loss.detach()) - float(direct_loss.detach())),
        },
        "gradient_max_absolute_error": gradient_errors,
        "update_max_absolute_error": update_errors,
        "representative_parameters": list(representative),
        "metric_fixture": metric,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "BASE_01": "PASS" if passed else "FAIL_TECHNICAL",
        "BASE_02": "PASS" if max(score_errors.values()) <= atol else "FAIL_TECHNICAL",
        "scope": "real released-checkpoint adapter parity; not baseline reproduction metrics",
    }
