"""Full-gallery P14T reproduction of the released CiCo checkpoint."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from ocem.baselines.cico_adapter import CiCoAdapter, CiCoEncoded
from ocem.evaluation.full_gallery import evaluate_full_gallery
from ocem.evaluation.scoring import (
    TextIndex,
    VideoIndex,
    score_directional_full_gallery,
)
from ocem.provenance.hashes import sha256_file


class CiCoReproductionError(ValueError):
    """Raised when reproduction inputs or execution violate a locked gate."""


def _verify(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise CiCoReproductionError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _load_json_lock(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    _verify(path, expected_sha256, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CiCoReproductionError(f"invalid {label}: {error}") from error
    if not isinstance(value, Mapping) or value.get("status") != "PASS":
        raise CiCoReproductionError(f"{label} is not PASS")
    return value


def _validate_gate_locks(
    protocol: Mapping[str, Any],
    features: Mapping[str, Any],
    expected_feature_lock_sha256: str,
) -> None:
    """Require a mutually linked P14T protocol/feature G0 pair."""

    if protocol.get("dataset") != "phoenix2014t" or features.get("dataset") != "phoenix2014t":
        raise CiCoReproductionError("locks do not identify phoenix2014t")
    if protocol.get("protocol_equivalence") != "PASS":
        raise CiCoReproductionError("protocol lock does not establish protocol equivalence")
    if protocol.get("feature_lock_sha256") != expected_feature_lock_sha256:
        raise CiCoReproductionError("protocol lock does not reference the supplied feature lock")
    if features.get("ready_for_dataset_g0") is not True:
        raise CiCoReproductionError("feature lock is not ready for the dataset G0 gate")
    if features.get("validation_or_test_used_for_training") is not False:
        raise CiCoReproductionError("feature lock does not prove train-only adaptation")


def _tokenize(tokenizer: Any, text: str, max_words: int = 32) -> tuple[list[int], list[bool]]:
    words = ["<|startoftext|>"] + tokenizer.tokenize(text)
    if len(words) > max_words - 1:
        indices = [0] + list(np.linspace(1, len(words) - 1, max_words - 2, dtype=int))
        words = list(np.asarray(words)[indices])
    words.append("<|endoftext|>")
    ids = tokenizer.convert_tokens_to_ids(words)
    valid = [True] * len(ids)
    ids.extend([0] * (max_words - len(ids)))
    valid.extend([False] * (max_words - len(valid)))
    if len(ids) != max_words:
        raise CiCoReproductionError("tokenized caption exceeds the locked input length")
    return ids, valid


def _selected_feature(path: Path, feature_len: int = 64) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - feature lock authenticates local cache
    feature = np.asarray(payload["feature"], dtype=np.float32)
    if feature.ndim != 2 or feature.shape[1] != 1024 or feature.shape[0] < 1:
        raise CiCoReproductionError(f"invalid feature shape at {path}: {feature.shape}")
    indices = (
        np.linspace(0, feature.shape[0] - 1, feature_len, dtype=int)
        if feature.shape[0] >= feature_len
        else np.arange(feature.shape[0], dtype=int)
    )
    selected = np.zeros((feature_len, 1024), dtype=np.float32)
    selected[: len(indices)] = feature[indices]
    valid = np.zeros(feature_len, dtype=bool)
    valid[: len(indices)] = True
    return selected, valid


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_modules(upstream_root: Path, expected: Mapping[str, str]) -> tuple[Any, Any, Any]:
    paths = {
        "modeling": upstream_root / "modules/modeling.py",
        "module_clip": upstream_root / "modules/module_clip.py",
        "tokenization": upstream_root / "modules/tokenization_clip.py",
        "metrics": upstream_root / "metrics.py",
    }
    for name, path in paths.items():
        if name not in expected:
            raise CiCoReproductionError(f"missing expected source hash for {name}")
        _verify(path, expected[name], name)
    resolved = str(upstream_root.resolve())
    sys.path.insert(0, resolved)
    try:
        modeling = importlib.import_module("modules.modeling")
        tokenization = importlib.import_module("modules.tokenization_clip")
        metrics = importlib.import_module("metrics")
    finally:
        if sys.path[0] == resolved:
            sys.path.pop(0)
    return modeling, tokenization, metrics


def reproduce_cico_phoenix2014t(
    *,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    clip_checkpoint: str | Path,
    expected_clip_sha256: str,
    upstream_root: str | Path,
    expected_source_hashes: Mapping[str, str],
    protocol_lock: str | Path,
    expected_protocol_lock_sha256: str,
    feature_lock: str | Path,
    expected_feature_lock_sha256: str,
    test_manifest: str | Path,
    cico_test_data: str | Path,
    expected_cico_test_data_sha256: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    encode_batch_size: int = 256,
    score_block_size: int = 128,
    alpha: float = 0.9,
) -> dict[str, Any]:
    """Run released-checkpoint inference on the locked official 642-pair gallery."""

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise CiCoReproductionError("full CiCo reproduction requires an available CUDA device")
    if encode_batch_size < 1 or score_block_size < 1 or not 0.0 <= alpha <= 1.0:
        raise CiCoReproductionError("invalid batch, block, or alpha setting")
    checkpoint, clip_checkpoint, upstream_root = (
        Path(checkpoint),
        Path(clip_checkpoint),
        Path(upstream_root),
    )
    protocol_lock, feature_lock = Path(protocol_lock), Path(feature_lock)
    test_manifest, cico_test_data, output_dir = (
        Path(test_manifest),
        Path(cico_test_data),
        Path(output_dir),
    )
    _verify(checkpoint, expected_checkpoint_sha256, "CiCo checkpoint")
    _verify(clip_checkpoint, expected_clip_sha256, "CLIP checkpoint")
    protocol = _load_json_lock(
        protocol_lock, expected_protocol_lock_sha256, "protocol lock"
    )
    features = _load_json_lock(feature_lock, expected_feature_lock_sha256, "feature lock")
    _validate_gate_locks(protocol, features, expected_feature_lock_sha256)
    streams = features.get("streams")
    if not isinstance(streams, Mapping):
        raise CiCoReproductionError("feature lock has no stream mapping")
    try:
        agnostic_root = Path(str(streams["domain_agnostic"]["root"])) / "test"
        adapted_root = Path(str(streams["domain_adapted_p14t"]["root"])) / "test"
    except (KeyError, TypeError) as error:
        raise CiCoReproductionError("feature lock does not contain both required streams") from error
    _verify(cico_test_data, expected_cico_test_data_sha256, "CiCo P14T test text data")
    modeling, tokenization, metrics_module = _source_modules(
        upstream_root, expected_source_hashes
    )

    manifest_records = [
        json.loads(line) for line in test_manifest.read_text(encoding="utf-8").splitlines()
    ]
    manifest_by_id = {str(record["sample_id"]): record for record in manifest_records}
    if len(manifest_by_id) != len(manifest_records):
        raise CiCoReproductionError("test manifest contains duplicate IDs")
    with cico_test_data.open("rb") as handle:
        text_data = pickle.load(handle)  # noqa: S301 - hash-pinned upstream artifact
    if not isinstance(text_data, Mapping) or set(text_data) != set(manifest_by_id):
        raise CiCoReproductionError("CiCo test text IDs differ from the protocol manifest")
    sample_ids = [str(sample_id) for sample_id in text_data]
    manifest_order = [str(record["sample_id"]) for record in manifest_records]
    order_equal = sample_ids == manifest_order
    if not order_equal:
        raise CiCoReproductionError("CiCo and manifest gallery orders differ")
    if any(str(text_data[key].get("video_name")) != key for key in sample_ids):
        raise CiCoReproductionError("CiCo video_name does not match its paired ID")
    if any(
        str(text_data[key].get("ori_text")) != str(manifest_by_id[key]["caption_raw"])
        for key in sample_ids
    ):
        raise CiCoReproductionError("CiCo ori_text differs from the official manifest caption")
    english_texts = [str(text_data[key].get("text", "")) for key in sample_ids]
    if any(not text.strip() for text in english_texts):
        raise CiCoReproductionError("CiCo test text contains a blank caption")

    tokenizer = tokenization.SimpleTokenizer()
    tokenized = [_tokenize(tokenizer, text) for text in english_texts]
    all_text_ids = torch.tensor([item[0] for item in tokenized], dtype=torch.long)
    all_text_valid = torch.tensor([item[1] for item in tokenized], dtype=torch.bool)
    resolved_device = torch.device(device)
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
        alpha=alpha,
        aug_choose="t2v",
        not_load_visual=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = modeling.CLIP4Clip.from_pretrained(
        "cross-base", cache_dir="", distributed=False, state_dict=state, task_config=config
    ).to(resolved_device)
    model.eval()
    adapter = CiCoAdapter(model)
    visual_tokens, video_valid, text_tokens, text_valid = [], [], [], []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(sample_ids), encode_batch_size):
            stop = min(start + encode_batch_size, len(sample_ids))
            fused, valid = [], []
            for sample_id in sample_ids[start:stop]:
                agnostic, agnostic_valid = _selected_feature(
                    agnostic_root / f"{sample_id}.pkl"
                )
                adapted, adapted_valid = _selected_feature(adapted_root / f"{sample_id}.pkl")
                if not np.array_equal(agnostic_valid, adapted_valid):
                    raise CiCoReproductionError(f"stream selection masks differ for {sample_id}")
                fused.append(alpha * agnostic + (1.0 - alpha) * adapted)
                valid.append(agnostic_valid)
            batch = {
                "local_h": torch.from_numpy(np.stack(fused)).to(resolved_device),
                "video_valid": torch.from_numpy(np.stack(valid)).to(resolved_device),
                "text_ids": all_text_ids[start:stop].to(resolved_device),
                "text_input_valid": all_text_valid[start:stop].to(resolved_device),
            }
            encoded = adapter.encode(batch, training=False)
            visual_tokens.append(encoded.visual_tokens.cpu())
            video_valid.append((encoded.upstream_video_mask == 0).cpu())
            text_tokens.append(encoded.canonical_text_tokens.cpu())
            text_valid.append((encoded.canonical_text_mask == 1).cpu())
            print(f"cico encode {stop}/{len(sample_ids)}", flush=True)
    visual_tokens_tensor = torch.cat(visual_tokens)
    video_valid_tensor = torch.cat(video_valid)
    text_tokens_tensor = torch.cat(text_tokens)
    text_valid_tensor = torch.cat(text_valid)
    dummy_intervals = torch.zeros(
        (*video_valid_tensor.shape, 2), dtype=torch.float64
    )
    video_index = VideoIndex(
        tuple(sample_ids), visual_tokens_tensor, video_valid_tensor, dummy_intervals
    )
    text_index = TextIndex(tuple(sample_ids), text_tokens_tensor, text_valid_tensor)

    def scorer(video, video_mask, intervals, text, canonical_mask):
        del intervals
        encoded = CiCoEncoded(
            visual_tokens=video.to(resolved_device),
            canonical_text_tokens=text.to(resolved_device),
            augmented_text_tokens=text.to(resolved_device),
            upstream_video_mask=(~video_mask).long().to(resolved_device),
            canonical_text_mask=canonical_mask.long().to(resolved_device),
            augmented_text_mask=canonical_mask.long().to(resolved_device),
        )
        raw = adapter.directional_matrices(encoded)
        return raw.raw_t2v, raw.raw_v2t

    with torch.no_grad():
        raw = score_directional_full_gallery(
            video_index,
            text_index,
            scorer,
            video_block_size=score_block_size,
            text_block_size=score_block_size,
        )
    scale = float(adapter.logit_scale().detach().cpu())
    legacy = scale * (
        adapter.dual_mix * raw.scores_t2v.numpy()
        + (1.0 - adapter.dual_mix) * raw.scores_v2t.numpy()
    )
    raw_t2v = raw.scores_t2v.numpy()
    raw_v2t = raw.scores_v2t.numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, value in (
        ("raw_t2v", raw_t2v),
        ("raw_v2t", raw_v2t),
        ("legacy_mixed_logits", legacy),
    ):
        path = output_dir / f"{name}.npy"
        _atomic_numpy(path, value.astype(np.float32, copy=False))
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "shape": list(value.shape),
            "dtype": "float32",
        }

    legacy_tensor = legacy[:, None, :]
    legacy_t2v = metrics_module.compute_metrics(
        metrics_module.tensor_video_to_text_sim(legacy_tensor.copy())
    )
    legacy_v2t = metrics_module.tensor_text_to_video_metrics(legacy_tensor.copy())
    mapping = {sample_id: sample_id for sample_id in sample_ids}
    corrected = evaluate_full_gallery(
        legacy,
        legacy,
        video_ids=sample_ids,
        text_ids=sample_ids,
        text_to_video=mapping,
        video_to_text=mapping,
    )
    published = {"T2V_R1": 69.5, "V2T_R1": 70.2}
    differences = {
        "T2V_R1_pp": float(legacy_t2v["R1"] - published["T2V_R1"]),
        "V2T_R1_pp": float(legacy_v2t["R1"] - published["V2T_R1"]),
    }
    gate_pass = all(abs(value) <= 1.0 for value in differences.values())
    counts = Counter(english_texts)
    return {
        "schema_version": "ocem.cico_baseline_reproduction.v1",
        "status": "PASS" if gate_pass else "FAIL_SCIENTIFIC",
        "G1": "PASS" if gate_pass else "FAIL_SCIENTIFIC",
        "dataset": "phoenix2014t",
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": expected_checkpoint_sha256},
        "clip_checkpoint": {
            "path": str(clip_checkpoint.resolve()),
            "sha256": expected_clip_sha256,
        },
        "protocol_lock": {
            "path": str(protocol_lock.resolve()),
            "sha256": expected_protocol_lock_sha256,
        },
        "feature_lock": {
            "path": str(feature_lock.resolve()),
            "sha256": expected_feature_lock_sha256,
        },
        "text_resource": {
            "path": str(cico_test_data.resolve()),
            "sha256": expected_cico_test_data_sha256,
            "field": "text",
            "language": "English translation supplied by pinned CiCo source",
            "ori_text_equals_manifest_caption_raw": True,
            "ids_and_order_equal_manifest": order_equal,
            "blank_captions": 0,
            "exact_duplicate_groups": sum(count > 1 for count in counts.values()),
            "exact_duplicate_rows": sum(count for count in counts.values() if count > 1),
        },
        "recipe": {
            "alpha": alpha,
            "fusion": "alpha*domain_agnostic + (1-alpha)*domain_adapted_p14t",
            "feature_len": 64,
            "max_words": 32,
            "encode_batch_size": encode_batch_size,
            "score_block_size": score_block_size,
            "dual_mix": adapter.dual_mix,
            "legacy_mix": "dual_mix*raw_t2v + (1-dual_mix)*raw_v2t",
            "external_logit_scale": scale,
        },
        "gallery": {
            "videos": len(sample_ids),
            "texts": len(sample_ids),
            "pairs_scored_per_direction": raw.scored_pairs,
            "expected_pairs_per_direction": raw.expected_pairs,
            "coverage": raw.scored_pairs / raw.expected_pairs,
            "sample_ids_sha256": hashlib.sha256(
                json.dumps(sample_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "scores": artifacts,
        "legacy_metrics": {"T2V": legacy_t2v, "V2T": legacy_v2t},
        "corrected_metrics": corrected,
        "published_R1": published,
        "difference_from_published": differences,
        "gate_tolerance_pp": 1.0,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": dict(expected_source_hashes),
        "validation_or_test_used_for_training": False,
    }
