"""Frozen CiCo Stage-A concentration diagnosis on the P14T validation gallery."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ocem.baselines.cico_adapter import CiCoAdapter, CiCoEncoded
from ocem.baselines.cico_reproduction import (
    _atomic_numpy,
    _feature_indices,
    _load_json_lock,
    _source_modules,
    _tokenize,
    _validate_gate_locks,
    _verify,
)
from ocem.diagnostics.concentration import (
    adjusted_paired_effect,
    bootstrap_mean_ci,
    independent_concentration,
)
from ocem.diagnostics.errors import equivalence_recall, select_r1_errors
from ocem.diagnostics.shortcuts import (
    assign_quantile_bin,
    caption_class_sizes,
    deterministic_negative_overlaps,
    lexical_jaccard,
    quantile_bin_edges,
)
from ocem.evaluation.scoring import TextIndex, VideoIndex, score_directional_full_gallery
from ocem.provenance.hashes import sha256_file
from ocem.scoring.geometry import build_support_geometry


class StageADiagnosticError(ValueError):
    """Raised when Stage A cannot be run under the preregistered contracts."""


def _load_manifest(path: Path, expected_split: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [str(record.get("sample_id", "")) for record in records]
    if not records or len(set(ids)) != len(ids):
        raise StageADiagnosticError(f"{expected_split} manifest has empty/duplicate IDs")
    if any(record.get("split") != expected_split for record in records):
        raise StageADiagnosticError(f"{expected_split} manifest contains another split")
    return records


def _load_text_mapping(path: Path, expected_sha256: str) -> Mapping[str, Mapping[str, Any]]:
    _verify(path, expected_sha256, f"CiCo text resource {path.name}")
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - resource hash is locked
    if not isinstance(value, Mapping):
        raise StageADiagnosticError(f"{path} is not a text mapping")
    return value


def _verify_text_records(
    records: Sequence[Mapping[str, Any]], text: Mapping[str, Mapping[str, Any]]
) -> None:
    ids = [str(record["sample_id"]) for record in records]
    if list(map(str, text)) != ids:
        raise StageADiagnosticError("CiCo text order differs from the declared manifests")
    by_id = {str(record["sample_id"]): record for record in records}
    for sample_id in ids:
        item = text[sample_id]
        if str(item.get("video_name")) != sample_id:
            raise StageADiagnosticError(f"text video_name mismatch for {sample_id}")
        if str(item.get("ori_text")) != str(by_id[sample_id]["caption_raw"]):
            raise StageADiagnosticError(f"text ori_text mismatch for {sample_id}")
        if not str(item.get("text", "")).strip():
            raise StageADiagnosticError(f"blank English text for {sample_id}")


def _selected_inputs(
    sample_id: str,
    *,
    agnostic_dir: Path,
    adapted_dir: Path,
    temporal_dir: Path,
    feature_len: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    original_count = None
    for root in (agnostic_dir, adapted_dir):
        path = root / f"{sample_id}.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - feature lock authenticates cache
        value = np.asarray(payload["feature"], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 1024 or value.shape[0] < 1:
            raise StageADiagnosticError(f"invalid feature shape for {sample_id}: {value.shape}")
        if original_count is None:
            original_count = value.shape[0]
        elif value.shape[0] != original_count:
            raise StageADiagnosticError(f"feature streams have different lengths for {sample_id}")
        features.append(value)
    assert original_count is not None
    indices = _feature_indices(original_count, feature_len)
    selected = np.zeros((feature_len, 1024), dtype=np.float32)
    selected[: len(indices)] = alpha * features[0][indices] + (1.0 - alpha) * features[1][indices]
    valid = np.zeros(feature_len, dtype=bool)
    valid[: len(indices)] = True
    metadata = json.loads((temporal_dir / f"{sample_id}.json").read_text(encoding="utf-8"))
    starts = np.asarray(metadata.get("rf_start"), dtype=np.float64)
    ends = np.asarray(metadata.get("rf_end"), dtype=np.float64)
    if starts.shape != (original_count,) or ends.shape != (original_count,):
        raise StageADiagnosticError(f"temporal supports do not align for {sample_id}")
    intervals = np.zeros((feature_len, 2), dtype=np.float64)
    intervals[: len(indices), 0] = starts[indices]
    intervals[: len(indices), 1] = ends[indices]
    if np.any(intervals[: len(indices), 1] <= intervals[: len(indices), 0]):
        raise StageADiagnosticError(f"invalid selected support for {sample_id}")
    return selected, valid, intervals


def _affinity(
    visual: torch.Tensor,
    video_valid: torch.Tensor,
    text: torch.Tensor,
    text_valid: torch.Tensor,
) -> np.ndarray:
    visual = visual[video_valid].to(dtype=torch.float64)
    text = text[text_valid].to(dtype=torch.float64)
    visual = visual / visual.norm(dim=-1, keepdim=True)
    text = text / text.norm(dim=-1, keepdim=True)
    value = text @ visual.T
    if not bool(torch.isfinite(value).all()):
        raise StageADiagnosticError("contextual token affinity is non-finite")
    return value.numpy()


def _pair_stats(
    *,
    visual: torch.Tensor,
    video_valid: torch.Tensor,
    intervals: torch.Tensor,
    text: torch.Tensor,
    text_valid: torch.Tensor,
    epsilon: float,
    null_prior: float,
    kappa: float,
) -> dict[str, float | int]:
    valid_intervals = intervals[video_valid].numpy()
    geometry = build_support_geometry(valid_intervals)
    return independent_concentration(
        _affinity(visual, video_valid, text, text_valid),
        geometry.A,
        geometry.w,
        geometry.q,
        epsilon=epsilon,
        null_prior=null_prior,
        kappa=kappa,
    ).to_dict()


def _model_config(clip_checkpoint: Path, alpha: float) -> SimpleNamespace:
    return SimpleNamespace(
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


def _encoded_score_block(
    video: torch.Tensor,
    video_valid: torch.Tensor,
    text: torch.Tensor,
    text_valid: torch.Tensor,
    device: torch.device,
) -> CiCoEncoded:
    """Reconstruct an encoded block without adding a second visual CLS mask."""

    if video.shape[:2] != video_valid.shape:
        raise StageADiagnosticError("contextual video tokens and validity mask do not align")
    return CiCoEncoded(
        visual_tokens=video.to(device),
        canonical_text_tokens=text.to(device),
        augmented_text_tokens=text.to(device),
        upstream_video_mask=(~video_valid).long().to(device),
        canonical_text_mask=text_valid.long().to(device),
        augmented_text_mask=text_valid.long().to(device),
    )


def _encode_batches(
    *,
    records: Sequence[Mapping[str, Any]],
    english: Mapping[str, str],
    tokenizer: Any,
    adapter: CiCoAdapter,
    agnostic_dir: Path,
    adapted_dir: Path,
    temporal_dir: Path,
    device: torch.device,
    batch_size: int,
    alpha: float,
):
    ids = [str(record["sample_id"]) for record in records]
    tokenized = {sample_id: _tokenize(tokenizer, english[sample_id]) for sample_id in ids}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            selected = [
                _selected_inputs(
                    sample_id,
                    agnostic_dir=agnostic_dir,
                    adapted_dir=adapted_dir,
                    temporal_dir=temporal_dir,
                    feature_len=64,
                    alpha=alpha,
                )
                for sample_id in batch_ids
            ]
            batch = {
                "local_h": torch.from_numpy(np.stack([item[0] for item in selected])).to(device),
                "video_valid": torch.from_numpy(np.stack([item[1] for item in selected])).to(device),
                "text_ids": torch.tensor(
                    [tokenized[sample_id][0] for sample_id in batch_ids], dtype=torch.long, device=device
                ),
                "text_input_valid": torch.tensor(
                    [tokenized[sample_id][1] for sample_id in batch_ids], dtype=torch.bool, device=device
                ),
            }
            encoded = adapter.encode(batch, training=False)
            padded_intervals = np.zeros((len(batch_ids), 65, 2), dtype=np.float64)
            padded_intervals[:, 1:] = np.stack([item[2] for item in selected])
            yield {
                "ids": batch_ids,
                "visual": encoded.visual_tokens.cpu(),
                "video_valid": (encoded.upstream_video_mask == 0).cpu(),
                "intervals": torch.from_numpy(padded_intervals),
                "text": encoded.Y.cpu(),
                "text_valid": encoded.text_valid.cpu(),
            }
            print(f"stage A encode {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def _subgroup_summaries(
    records: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for field in (
        "direction",
        "duration_bin",
        "window_bin",
        "text_token_bin",
        "lexical_bin",
        "signer_id",
    ):
        groups: dict[str, list[float]] = defaultdict(list)
        for record in records:
            groups[str(record[field])].append(float(record["concentration_difference"]))
        summaries = {}
        for key, values in sorted(groups.items()):
            summaries[key] = (
                bootstrap_mean_ci(values, replicates=replicates, seed=seed)
                if len(values) >= 2
                else {"pairs": 1, "mean": values[0], "ci95_lower": None, "ci95_upper": None}
            )
        result[field] = summaries
    return result


def diagnose_cico_stage_a(
    *,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    clip_checkpoint: str | Path,
    expected_clip_sha256: str,
    upstream_root: str | Path,
    expected_source_hashes: Mapping[str, str],
    baseline_lock: str | Path,
    expected_baseline_lock_sha256: str,
    protocol_lock: str | Path,
    expected_protocol_lock_sha256: str,
    feature_lock: str | Path,
    expected_feature_lock_sha256: str,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    train_text: str | Path,
    expected_train_text_sha256: str,
    dev_text: str | Path,
    expected_dev_text_sha256: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    encode_batch_size: int = 256,
    score_block_size: int = 128,
    alpha: float = 0.9,
    epsilon: float = 0.05,
    null_prior: float = 0.15,
    kappa: float = 1.5,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260910,
) -> dict[str, Any]:
    """Run the preregistered frozen contextual-token Stage-A gate."""

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise StageADiagnosticError("Stage A requires an available CUDA device")
    if encode_batch_size < 1 or score_block_size < 1 or not 0 <= alpha <= 1:
        raise StageADiagnosticError("invalid batch sizes or alpha")
    if bootstrap_replicates < 10_000:
        raise StageADiagnosticError("Stage A requires at least 10,000 bootstrap replicates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, clip_checkpoint, upstream_root = Path(checkpoint), Path(clip_checkpoint), Path(upstream_root)
    protocol_lock, feature_lock, baseline_lock = Path(protocol_lock), Path(feature_lock), Path(baseline_lock)
    _verify(checkpoint, expected_checkpoint_sha256, "CiCo checkpoint")
    _verify(clip_checkpoint, expected_clip_sha256, "CLIP checkpoint")
    protocol = _load_json_lock(protocol_lock, expected_protocol_lock_sha256, "protocol lock")
    features = _load_json_lock(feature_lock, expected_feature_lock_sha256, "feature lock")
    _validate_gate_locks(protocol, features, expected_feature_lock_sha256)
    baseline = _load_json_lock(baseline_lock, expected_baseline_lock_sha256, "baseline lock")
    if baseline.get("G1") != "PASS":
        raise StageADiagnosticError("baseline G1 is not PASS")
    baseline_inputs = baseline.get("inputs")
    if not isinstance(baseline_inputs, Mapping) or (
        baseline_inputs.get("cico_checkpoint_sha256") != expected_checkpoint_sha256
        or baseline_inputs.get("feature_lock_sha256") != expected_feature_lock_sha256
    ):
        raise StageADiagnosticError("baseline lock does not match checkpoint/feature lock")
    streams = features.get("streams")
    if not isinstance(streams, Mapping):
        raise StageADiagnosticError("feature lock has no streams")
    agnostic_root = Path(str(streams["domain_agnostic"]["root"]))
    adapted_root = Path(str(streams["domain_adapted_p14t"]["root"]))
    temporal_root = Path(str(features["temporal_root"]))

    train_records = _load_manifest(Path(train_manifest), "train")
    validation_records = _load_manifest(Path(validation_manifest), "validation")
    if {str(item["sample_id"]) for item in train_records} & {
        str(item["sample_id"]) for item in validation_records
    }:
        raise StageADiagnosticError("train and validation manifests overlap")
    train_mapping = _load_text_mapping(Path(train_text), expected_train_text_sha256)
    _verify_text_records(train_records, train_mapping)
    dev_mapping = _load_text_mapping(Path(dev_text), expected_dev_text_sha256)
    combined = [*train_records, *validation_records]
    _verify_text_records(combined, dev_mapping)
    english_train = {str(item["sample_id"]): str(train_mapping[str(item["sample_id"])]["text"]) for item in train_records}
    english_validation = {
        str(item["sample_id"]): str(dev_mapping[str(item["sample_id"])]["text"])
        for item in validation_records
    }

    modeling, tokenization, _ = _source_modules(upstream_root, expected_source_hashes)
    config = _model_config(clip_checkpoint, alpha)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resolved_device = torch.device(device)
    model = modeling.CLIP4Clip.from_pretrained(
        "cross-base", cache_dir="", distributed=False, state_dict=state, task_config=config
    ).to(resolved_device)
    model.eval()
    adapter = CiCoAdapter(model)
    tokenizer = tokenization.SimpleTokenizer()

    train_by_id = {str(record["sample_id"]): record for record in train_records}
    train_positive = []
    for batch in _encode_batches(
        records=train_records,
        english=english_train,
        tokenizer=tokenizer,
        adapter=adapter,
        agnostic_dir=agnostic_root / "train",
        adapted_dir=adapted_root / "train",
        temporal_dir=temporal_root / "train",
        device=resolved_device,
        batch_size=encode_batch_size,
        alpha=alpha,
    ):
        for index, sample_id in enumerate(batch["ids"]):
            stats = _pair_stats(
                visual=batch["visual"][index, 1:],
                video_valid=batch["video_valid"][index, 1:],
                intervals=batch["intervals"][index, 1:],
                text=batch["text"][index],
                text_valid=batch["text_valid"][index],
                epsilon=epsilon,
                null_prior=null_prior,
                kappa=kappa,
            )
            record = train_by_id[sample_id]
            train_positive.append(
                {
                    "sample_id": sample_id,
                    **stats,
                    "duration_s": float(record["end_time_s"]) - float(record["start_time_s"]),
                    "selected_windows": int(batch["video_valid"][index, 1:].sum()),
                    "text_tokens": int(batch["text_valid"][index].sum()),
                    "caption_sha256": str(record["caption_sha256"]),
                    "signer_id": record.get("signer_id"),
                }
            )
    threshold = float(
        np.quantile(
            [item["concentration_p95"] for item in train_positive],
            0.90,
            method="linear",
        )
    )
    train_captions = {str(record["sample_id"]): str(record["caption_raw"]) for record in train_records}
    bin_edges = {
        "duration_s": quantile_bin_edges([item["duration_s"] for item in train_positive]),
        "selected_windows": quantile_bin_edges([item["selected_windows"] for item in train_positive]),
        "text_tokens": quantile_bin_edges([item["text_tokens"] for item in train_positive]),
        "lexical_overlap": quantile_bin_edges(
            deterministic_negative_overlaps(
                [str(record["sample_id"]) for record in train_records],
                train_captions,
                seed=bootstrap_seed,
            )
        ),
    }
    _write_jsonl(output_dir / "train_positive_stats.jsonl", train_positive)

    validation_batches = list(
        _encode_batches(
            records=validation_records,
            english=english_validation,
            tokenizer=tokenizer,
            adapter=adapter,
            agnostic_dir=agnostic_root / "dev",
            adapted_dir=adapted_root / "dev",
            temporal_dir=temporal_root / "dev",
            device=resolved_device,
            batch_size=encode_batch_size,
            alpha=alpha,
        )
    )
    ids = [sample_id for batch in validation_batches for sample_id in batch["ids"]]
    visual = torch.cat([batch["visual"] for batch in validation_batches])
    video_valid = torch.cat([batch["video_valid"] for batch in validation_batches])
    intervals = torch.cat([batch["intervals"] for batch in validation_batches])
    text_tokens = torch.cat([batch["text"] for batch in validation_batches])
    text_valid = torch.cat([batch["text_valid"] for batch in validation_batches])
    video_index = VideoIndex(tuple(ids), visual, video_valid, intervals)
    text_index = TextIndex(tuple(ids), text_tokens, text_valid)

    def scorer(video, video_mask, unused_intervals, text, canonical_mask):
        del unused_intervals
        encoded = _encoded_score_block(
            video,
            video_mask,
            text,
            canonical_mask,
            resolved_device,
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
    raw_t2v, raw_v2t = raw.scores_t2v.numpy(), raw.scores_v2t.numpy()
    legacy = float(adapter.logit_scale().detach().cpu()) * (
        adapter.dual_mix * raw_t2v + (1.0 - adapter.dual_mix) * raw_v2t
    )
    for name, value in (("raw_t2v.validation", raw_t2v), ("raw_v2t.validation", raw_v2t), ("legacy_mixed_logits.validation", legacy)):
        _atomic_numpy(output_dir / f"{name}.npy", value.astype(np.float32, copy=False))

    validation_by_id = {str(record["sample_id"]): record for record in validation_records}
    id_position = {sample_id: index for index, sample_id in enumerate(ids)}
    hash_by_id = {sample_id: str(validation_by_id[sample_id]["caption_sha256"]) for sample_id in ids}
    class_size = caption_class_sizes(list(hash_by_id.values()))
    errors = [
        *select_r1_errors(legacy, video_ids=ids, text_ids=ids, direction="T2V"),
        *select_r1_errors(legacy, video_ids=ids, text_ids=ids, direction="V2T"),
    ]
    pair_records = []
    for error in errors:
        query_id = str(error["query_id"])
        true_video_id, false_video_id = str(error["true_video_id"]), str(error["false_video_id"])
        true_text_id, false_text_id = str(error["true_text_id"]), str(error["false_text_id"])

        def stats(video_id: str, text_id: str):
            vi, ti = id_position[video_id], id_position[text_id]
            return _pair_stats(
                visual=visual[vi, 1:],
                video_valid=video_valid[vi, 1:],
                intervals=intervals[vi, 1:],
                text=text_tokens[ti],
                text_valid=text_valid[ti],
                epsilon=epsilon,
                null_prior=null_prior,
                kappa=kappa,
            )

        true_stats, false_stats = stats(true_video_id, true_text_id), stats(false_video_id, false_text_id)

        def controls(
            video_id: str, text_id: str, candidate_caption_id: str
        ) -> dict[str, float | int]:
            video_record = validation_by_id[video_id]
            duration = float(video_record["end_time_s"]) - float(video_record["start_time_s"])
            return {
                "duration_s": duration,
                "selected_windows": int(video_valid[id_position[video_id], 1:].sum()),
                "text_tokens": int(text_valid[id_position[text_id]].sum()),
                "lexical_overlap": lexical_jaccard(
                    str(validation_by_id[query_id]["caption_raw"]),
                    str(validation_by_id[candidate_caption_id]["caption_raw"]),
                ),
                "caption_class_size": class_size[hash_by_id[candidate_caption_id]],
            }

        true_candidate_caption_id = true_video_id if error["direction"] == "T2V" else true_text_id
        false_candidate_caption_id = false_video_id if error["direction"] == "T2V" else false_text_id
        true_controls = controls(true_video_id, true_text_id, true_candidate_caption_id)
        false_controls = controls(false_video_id, false_text_id, false_candidate_caption_id)
        ambiguity = hash_by_id[query_id] == hash_by_id[false_candidate_caption_id]
        difference = float(false_stats["concentration_p95"] - true_stats["concentration_p95"])
        affected = bool(
            not ambiguity
            and false_stats["real_mass"] >= 0.1
            and difference > 0
            and false_stats["concentration_p95"] > threshold
            and false_stats["excess_kappa"] > true_stats["excess_kappa"]
        )
        false_duration = float(false_controls["duration_s"])
        false_windows = float(false_controls["selected_windows"])
        false_text_count = float(false_controls["text_tokens"])
        false_lexical = float(false_controls["lexical_overlap"])
        pair_records.append(
            {
                **error,
                "exact_caption_ambiguity": bool(ambiguity),
                "affected": affected,
                "concentration_difference": difference,
                "true": {"stats": true_stats, "controls": true_controls},
                "false": {"stats": false_stats, "controls": false_controls},
                "duration_bin": assign_quantile_bin(false_duration, bin_edges["duration_s"]),
                "window_bin": assign_quantile_bin(false_windows, bin_edges["selected_windows"]),
                "text_token_bin": assign_quantile_bin(false_text_count, bin_edges["text_tokens"]),
                "lexical_bin": assign_quantile_bin(false_lexical, bin_edges["lexical_overlap"]),
                "signer_id": validation_by_id[query_id].get("signer_id"),
            }
        )
    _write_jsonl(output_dir / "validation_error_pairs.jsonl", pair_records)

    primary = [record for record in pair_records if not record["exact_caption_ambiguity"]]
    if len(primary) < 2:
        raise StageADiagnosticError("fewer than two non-ambiguous validation errors")
    differences = [float(record["concentration_difference"]) for record in primary]
    ordinary_ci = bootstrap_mean_ci(
        differences, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    caption_cluster_ci = bootstrap_mean_ci(
        differences,
        group_ids=[hash_by_id[str(record["query_id"])] for record in primary],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1,
    )
    controls = []
    for record in primary:
        true_control, false_control = record["true"]["controls"], record["false"]["controls"]
        controls.append(
            [
                math.log1p(float(false_control["duration_s"])) - math.log1p(float(true_control["duration_s"])),
                float(false_control["selected_windows"]) - float(true_control["selected_windows"]),
                float(false_control["text_tokens"]) - float(true_control["text_tokens"]),
                float(false_control["lexical_overlap"]) - float(true_control["lexical_overlap"]),
                math.log1p(float(false_control["caption_class_size"]))
                - math.log1p(float(true_control["caption_class_size"])),
            ]
        )
    adjusted = adjusted_paired_effect(
        differences,
        np.asarray(controls),
        ["log_duration", "selected_windows", "text_tokens", "lexical_overlap", "log_caption_class_size"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 2,
    )
    per_direction = {
        direction: bootstrap_mean_ci(
            [float(record["concentration_difference"]) for record in primary if record["direction"] == direction],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 3 + index,
        )
        for index, direction in enumerate(("T2V", "V2T"))
    }
    affected_count = sum(bool(record["affected"]) for record in pair_records)
    affected_fraction = affected_count / len(pair_records)
    checks = {
        "paired_ci_above_zero": ordinary_ci["ci95_lower"] > 0,
        "caption_cluster_ci_above_zero": caption_cluster_ci["ci95_lower"] > 0,
        "adjusted_ci_above_zero": adjusted["ci95_lower"] > 0,
        "both_direction_means_positive": all(item["mean"] > 0 for item in per_direction.values()),
        "affected_fraction_at_least_10pct": affected_fraction >= 0.10,
    }
    passed = all(checks.values())
    artifacts = {}
    for name in (
        "train_positive_stats.jsonl",
        "validation_error_pairs.jsonl",
        "raw_t2v.validation.npy",
        "raw_v2t.validation.npy",
        "legacy_mixed_logits.validation.npy",
    ):
        path = output_dir / name
        artifacts[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    duplicate_control = {
        direction: equivalence_recall(
            legacy,
            video_ids=ids,
            text_ids=ids,
            caption_hash_by_id=hash_by_id,
            direction=direction,
        )
        for direction in ("T2V", "V2T")
    }
    return {
        "schema_version": "ocem.failure_gate_A.v1",
        "status": "PASS" if passed else "NO_GO_SCIENTIFIC",
        "G2": "PASS" if passed else "NO_GO_SCIENTIFIC",
        "stage": "A",
        "interpretation_scope": "contextual-token index proxy; not physical local-support evidence",
        "preregistered_statistic": {
            "plan": "independent closed form with shared q/null/epsilon and no capacity constraint",
            "primary": "left-continuous weighted P95 atom density",
            "train_positive_threshold_quantile": 0.90,
            "train_positive_threshold": threshold,
            "epsilon": epsilon,
            "null_prior": null_prior,
            "kappa_for_excess": kappa,
            "affected_rule": "non-ambiguous; false real_mass>=0.1; false concentration>true; false concentration>train-positive P90; false E_kappa>true",
        },
        "data": {
            "train_samples": len(train_records),
            "validation_samples": len(validation_records),
            "test_features_or_text_read": False,
            "train_validation_overlap": 0,
            "bin_edges_from_train": bin_edges,
        },
        "baseline_errors": {
            "T2V": len([record for record in pair_records if record["direction"] == "T2V"]),
            "V2T": len([record for record in pair_records if record["direction"] == "V2T"]),
            "total": len(pair_records),
            "non_ambiguous_primary": len(primary),
            "exact_caption_ambiguity": len(pair_records) - len(primary),
        },
        "association": {
            "paired": ordinary_ci,
            "caption_cluster_sensitivity": caption_cluster_ci,
            "adjusted": adjusted,
            "per_direction": per_direction,
            "subgroups": _subgroup_summaries(
                primary, replicates=bootstrap_replicates, seed=bootstrap_seed + 10
            ),
        },
        "affected_stratum": {
            "count": affected_count,
            "fraction_of_all_R1_errors": affected_fraction,
            "minimum_fraction": 0.10,
        },
        "duplicate_control": duplicate_control,
        "gate_checks": checks,
        "gallery": {
            "videos": len(ids),
            "texts": len(ids),
            "pairs_per_direction": raw.expected_pairs,
            "coverage": raw.scored_pairs / raw.expected_pairs,
            "sample_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        },
        "locks": {
            "baseline": {"path": str(baseline_lock.resolve()), "sha256": expected_baseline_lock_sha256},
            "protocol": {"path": str(protocol_lock.resolve()), "sha256": expected_protocol_lock_sha256},
            "features": {"path": str(feature_lock.resolve()), "sha256": expected_feature_lock_sha256},
        },
        "artifacts": artifacts,
        "source_sha256": dict(expected_source_hashes),
    }
