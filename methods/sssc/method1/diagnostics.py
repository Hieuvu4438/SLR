from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .auxiliary_cache import Method1AuxiliaryCache
from .baseline import baseline_loss_from_local, encode_local
from .config import Method1Config
from .data import Method1Collator, Method1Dataset, _load_records
from .distributed import DistributedRuntime
from .losses.shared_support import reference_support, span_contrast_terms
from .miner import construct_caption_edits
from .model_factory import build_upret_model, load_exact_student_state
from .reference import (
    encode_reference_text,
    encode_reference_video,
    freeze_reference,
    pool_reference_spans,
)
from .reference_pipeline import _video_batch, reference_cache_identity
from .schemas import GroupRecord, TextRecord, VideoRecord
from .token_spans import TokenizedCaption, tokenize_with_spans
from .upstream import create_upret_tokenizer
from .utils import atomic_json_dump, atomic_jsonl_dump, sha256_file, sha256_json, stable_seed


class DiagnosticError(RuntimeError):
    pass


def loss_gradient_cosine(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    base_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
) -> dict[str, Any]:
    parameters = [parameter for _, parameter in named_parameters if parameter.requires_grad]
    names = [name for name, parameter in named_parameters if parameter.requires_grad]
    if not parameters:
        raise DiagnosticError("gradient probe received no trainable shared visual parameters")
    base_gradients = torch.autograd.grad(
        base_loss, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss, parameters, allow_unused=True
    )
    dot = torch.zeros((), dtype=torch.float64, device=base_loss.device)
    base_squared = torch.zeros_like(dot)
    auxiliary_squared = torch.zeros_like(dot)
    jointly_active = 0
    for base_gradient, auxiliary_gradient in zip(
        base_gradients, auxiliary_gradients, strict=True
    ):
        if base_gradient is not None:
            base_squared += base_gradient.double().square().sum()
        if auxiliary_gradient is not None:
            auxiliary_squared += auxiliary_gradient.double().square().sum()
        if base_gradient is not None and auxiliary_gradient is not None:
            dot += (base_gradient.double() * auxiliary_gradient.double()).sum()
            if bool(base_gradient.abs().sum() > 0) and bool(
                auxiliary_gradient.abs().sum() > 0
            ):
                jointly_active += 1
    base_norm = base_squared.sqrt()
    auxiliary_norm = auxiliary_squared.sqrt()
    denominator = base_norm * auxiliary_norm
    cosine = float(dot / denominator) if bool(denominator > 0) else None
    return {
        "parameter_scope": "trainable_clip_visual",
        "parameter_tensor_count": len(parameters),
        "jointly_active_parameter_tensor_count": jointly_active,
        "parameter_names_sha256": sha256_json(names),
        "base_gradient_norm": float(base_norm),
        "auxiliary_gradient_norm": float(auxiliary_norm),
        "base_auxiliary_gradient_cosine": cosine,
    }
def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _load_frozen_candidate_graph(cache_root: Path) -> dict[str, tuple[tuple[str, float], ...]]:
    path = cache_root / "auxiliary" / "mining" / "candidate_graph.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"cannot read frozen train candidate graph: {error}") from error
    output = {}
    for word, candidates in raw.items():
        output[str(word)] = tuple(
            (str(item["replacement"]), float(item["cosine"])) for item in candidates
        )
    return output


def _encode_span_vectors(
    reference: torch.nn.Module,
    tokenizer: Any,
    captions: Sequence[tuple[str, str, tuple[int, ...]]],
    *,
    max_positions: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for start in range(0, len(captions), batch_size):
        current = captions[start : start + batch_size]
        encoded = [
            tokenize_with_spans(
                text,
                text_uid=text_uid,
                tokenizer=tokenizer,
                max_positions=max_positions,
            )
            for text_uid, text, _ in current
        ]
        ids = torch.tensor([item.input_ids for item in encoded], dtype=torch.long, device=device)
        valid = torch.tensor([item.text_valid for item in encoded], dtype=torch.bool, device=device)
        tokens, _ = encode_reference_text(reference, ids, torch.zeros_like(ids), valid)
        vectors = pool_reference_spans(
            reference, tokens, [positions for _, _, positions in current]
        ).cpu().numpy()
        output.extend(vector.astype(np.float32, copy=False) for vector in vectors)
    return output


def _support_metrics(
    x_ref: np.ndarray,
    video_valid: np.ndarray,
    selected_feature_indices: Sequence[int],
    q_positive: np.ndarray,
    q_negative: np.ndarray,
    *,
    tau: float,
    margin: float,
) -> dict[str, Any]:
    x = torch.from_numpy(np.asarray(x_ref, dtype=np.float32))[None]
    valid = torch.from_numpy(np.asarray(video_valid, dtype=np.bool_))[None]
    positive = torch.from_numpy(np.asarray(q_positive, dtype=np.float32))[None, None, None]
    negative = torch.from_numpy(np.asarray(q_negative, dtype=np.float32))[None, None, None]
    positive_support = reference_support(x, positive, valid, tau)[0, 0, 0]
    negative_support = reference_support(x, negative, valid, tau)[0, 0, 0]
    midpoint = 0.5 * (positive_support + negative_support)
    jsd = 0.5 * (
        torch.sum(
            torch.where(
                positive_support > 0,
                positive_support * torch.log(positive_support / midpoint.clamp_min(1e-12)),
                torch.zeros_like(positive_support),
            )
        )
        + torch.sum(
            torch.where(
                negative_support > 0,
                negative_support * torch.log(negative_support / midpoint.clamp_min(1e-12)),
                torch.zeros_like(negative_support),
            )
        )
    )
    valid_count = int(valid.sum())
    positive_entropy = -torch.sum(
        positive_support[valid[0]] * torch.log(positive_support[valid[0]] + 1e-12)
    )
    normalized_entropy = (
        float(positive_entropy / math.log(valid_count)) if valid_count > 1 else 0.0
    )
    positive_peak = int(torch.argmax(positive_support))
    negative_peak = int(torch.argmax(negative_support))
    selected = np.asarray(selected_feature_indices, dtype=np.int64)
    shared_delta = torch.einsum(
        "l,ld,d->", positive_support, x[0], positive[0, 0, 0] - negative[0, 0, 0]
    )
    independent_delta = torch.einsum(
        "l,ld,d->", positive_support, x[0], positive[0, 0, 0]
    ) - torch.einsum("l,ld,d->", negative_support, x[0], negative[0, 0, 0])
    return {
        "q_difference_norm": float(np.linalg.norm(q_positive - q_negative)),
        "text_vector_cosine": float(np.dot(q_positive, q_negative)),
        "support_overlap": float(torch.minimum(positive_support, negative_support).sum()),
        "support_jsd": float(jsd),
        "positive_support_entropy": normalized_entropy,
        "positive_support_concentration": 1.0 - normalized_entropy,
        "positive_peak_feature_index": int(selected[positive_peak]),
        "negative_peak_feature_index": int(selected[negative_peak]),
        "peak_displacement_feature_rows": int(
            abs(selected[positive_peak] - selected[negative_peak])
        ),
        "shared_margin": float(shared_delta),
        "independent_margin": float(independent_delta),
        "shared_hinge_active": bool(shared_delta < margin),
        "independent_hinge_active": bool(independent_delta < margin),
        "valid_clip_count": valid_count,
    }


def _feature_removal_positions(
    x_ref: np.ndarray,
    video_valid: np.ndarray,
    q_positives: Sequence[np.ndarray],
    selected_feature_indices: Sequence[int],
    *,
    tau: float,
    seed_parts: Sequence[Any],
) -> dict[str, Any]:
    valid_positions = np.flatnonzero(np.asarray(video_valid, dtype=np.bool_))
    selected_rows = np.asarray(selected_feature_indices, dtype=np.int64)
    if len(valid_positions) < 4 or not q_positives:
        return {
            "informative": False,
            "removal_count": 0,
            "selected_positions": (),
            "random_positions": (),
            "selected_feature_rows": (),
            "random_feature_rows": (),
        }
    x = torch.from_numpy(np.asarray(x_ref, dtype=np.float32))[None]
    valid = torch.from_numpy(np.asarray(video_valid, dtype=np.bool_))[None]
    supports = []
    for vector in q_positives:
        q_positive = torch.from_numpy(np.asarray(vector, dtype=np.float32))[None, None, None]
        supports.append(reference_support(x, q_positive, valid, tau)[0, 0, 0])
    average_support = torch.stack(supports).mean(dim=0).numpy()
    removal_count = min(len(valid_positions) - 1, max(1, math.floor(0.2 * len(valid_positions))))
    ranked = sorted(valid_positions.tolist(), key=lambda position: (-average_support[position], position))
    selected_positions = tuple(sorted(ranked[:removal_count]))
    rng = np.random.default_rng(stable_seed(*seed_parts, "feature_removal_random"))
    random_positions = tuple(
        sorted(int(position) for position in rng.choice(valid_positions, removal_count, replace=False))
    )
    if random_positions == selected_positions:
        order = valid_positions.tolist()
        shifted = {
            order[(order.index(position) + 1) % len(order)] for position in selected_positions
        }
        random_positions = tuple(sorted(shifted))
    return {
        "informative": True,
        "removal_count": removal_count,
        "selected_positions": selected_positions,
        "random_positions": random_positions,
        "selected_feature_rows": tuple(int(selected_rows[position]) for position in selected_positions),
        "random_feature_rows": tuple(int(selected_rows[position]) for position in random_positions),
    }


def _corrupt_and_reencode(
    reference: torch.nn.Module,
    video_features: torch.Tensor,
    video_ignore_raw: torch.Tensor,
    removals: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    features = video_features.repeat(len(removals), 1, 1, 1)
    masks = video_ignore_raw.repeat(len(removals), 1)
    for row, positions in enumerate(removals):
        if positions:
            position_tensor = torch.tensor(positions, dtype=torch.long, device=features.device)
            features[row, :, position_tensor, :] = 0.0
            masks[row, position_tensor + 1] = True
    return encode_reference_video(
        reference,
        {"video_features": features, "video_ignore_raw": masks},
    )


def _gradient_probe(
    config: Method1Config,
    *,
    upret_root: str | Path,
    device: torch.device,
    tokenizer: Any,
    auxiliary_cache: Method1AuxiliaryCache,
) -> dict[str, Any]:
    student, _ = build_upret_model(config, upret_root=upret_root)
    checkpoint = load_exact_student_state(student, config.reference.checkpoint)
    if not checkpoint.get("training_run_complete", False):
        raise DiagnosticError("gradient probe requires the completed selected baseline")
    student = student.to(device).train()
    dataset = Method1Dataset(
        config,
        split="train",
        tokenizer=tokenizer,
        augment=True,
        auxiliary_cache=auxiliary_cache,
    )
    probe_items = []
    for index in range(len(dataset)):
        item = dataset[index]
        if bool(item["edit_valid"].any()):
            probe_items.append(item)
        if len(probe_items) == 4:
            break
    if len(probe_items) < 2:
        raise DiagnosticError("gradient probe needs two train groups with eligible cached edits")
    batch = Method1Collator()(probe_items)
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    probe_seed = stable_seed(config.seed, "gradient_probe_v1") % (2**63 - 1)
    torch.manual_seed(probe_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(probe_seed)
    encoding = encode_local(student, batch)
    base = baseline_loss_from_local(
        student,
        encoding,
        runtime=DistributedRuntime(),
        seed=config.seed,
        optimizer_step=0,
        microstep=0,
        temperature=config.model.inner_similarity_temperature,
        checkpoint_score_blocks=False,
    )
    terms = span_contrast_terms(
        encoding.video_raw[:, 1:, :],
        batch["x_ref"],
        batch["q_pos"],
        batch["q_neg"],
        ~encoding.video_ignore_raw[:, 1:].bool(),
        batch["edit_valid"],
        torch.ones_like(batch["confidence"]),
        mode="shared",
        tau=config.auxiliary.tau_support,
        margin=config.auxiliary.margin,
    )
    if not bool(terms.denominator > 0):
        raise DiagnosticError("gradient probe selected no valid auxiliary weight")
    auxiliary = terms.numerator / terms.denominator
    visual_parameters = [
        (name, parameter)
        for name, parameter in student.named_parameters()
        if name.startswith("clip.visual.")
    ]
    report = loss_gradient_cosine(visual_parameters, base, auxiliary)
    report.update(
        {
            "status": "complete",
            "world_size": 1,
            "student_copy": "separate_unwrapped_selected_baseline",
            "fixed_noise": True,
            "probe_seed": probe_seed,
            "probe_video_uids": list(batch["video_uid"]),
            "probe_text_uids": list(batch["text_uid"]),
            "valid_edit_count": int(batch["edit_valid"].sum()),
            "scope_note": (
                "matched small-batch conflict probe; not an estimate of the global training gradient"
            ),
        }
    )
    return report


def support_diagnostics(
    config: Method1Config,
    *,
    split: str,
    upret_root: str | Path = "third_party/UPRet",
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    if split != "dev":
        raise DiagnosticError("support diagnostics are dev-only before configuration lock")
    cache_root = Path(config.reference.cache_dir)
    identity = reference_cache_identity(config)
    # This performs the full train-cache/miner/negative-cache identity audit without reusing
    # any train example as a dev observation.
    auxiliary_cache = Method1AuxiliaryCache(cache_root, expected_identity=identity)
    graph = _load_frozen_candidate_graph(cache_root)
    model, _ = build_upret_model(config, upret_root=upret_root)
    checkpoint = load_exact_student_state(model, config.reference.checkpoint)
    if (
        checkpoint.get("arm") != "base_initial"
        or checkpoint.get("dev_selection") is None
        or not checkpoint.get("training_run_complete", False)
    ):
        raise DiagnosticError("diagnostics require the completed selected baseline reference")
    torch_device = torch.device(device)
    reference = freeze_reference(model.to(torch_device))
    tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
    manifest_root = Path(config.data.manifest_dir)
    texts = {
        record.text_uid: record
        for record in _load_records(manifest_root / "texts.jsonl", TextRecord)
    }
    groups = [
        record
        for record in _load_records(manifest_root / "groups.jsonl", GroupRecord)
        if record.group_uid.startswith(f"{config.data.dataset}:{split}:")
    ]
    videos = {
        record.video_uid: record
        for record in _load_records(manifest_root / "videos.jsonl", VideoRecord)
        if record.split == split
    }
    train_hashes = {record.caption_hash for record in texts.values() if record.split == "train"}
    split_caption_counts: dict[str, int] = {}
    for group in groups:
        caption_hash = texts[group.text_uid].caption_hash
        split_caption_counts[caption_hash] = split_caption_counts.get(caption_hash, 0) + 1
    tokenized: dict[str, TokenizedCaption] = {
        group.text_uid: tokenize_with_spans(
            texts[group.text_uid].raw_text,
            text_uid=group.text_uid,
            tokenizer=tokenizer,
            max_positions=config.data.text_max_positions,
        )
        for group in groups
    }
    mined = construct_caption_edits(
        tokenized,
        graph,
        tokenizer=tokenizer,
        original_train_caption_hashes=train_hashes,
        max_edits_per_caption=config.miner.cached_negatives_per_caption,
        miner_version=f"{config.miner.version}:dev_diagnostic_only",
        max_positions=config.data.text_max_positions,
    )
    edits = [
        edit
        for text_uid in sorted(mined.edits_by_text_uid)
        for edit in mined.edits_by_text_uid[text_uid]
    ]
    if not edits:
        raise DiagnosticError("frozen train vocabulary produced no eligible dev diagnostic edits")
    text_inputs = []
    for edit in edits:
        positive = tokenized[edit.text_uid]
        text_inputs.append(
            (edit.text_uid, positive.canonical_text, edit.positive_token_positions)
        )
        text_inputs.append(
            (edit.text_uid, edit.negative_canonical_text, edit.negative_token_positions)
        )
    vectors = _encode_span_vectors(
        reference,
        tokenizer,
        text_inputs,
        max_positions=config.data.text_max_positions,
        batch_size=config.evaluation.encode_batch_size,
        device=torch_device,
    )
    edit_vectors = {
        edit.edit_uid: (vectors[2 * index], vectors[2 * index + 1])
        for index, edit in enumerate(edits)
    }
    per_example: list[dict[str, Any]] = []
    edits_by_text = mined.edits_by_text_uid
    for group in groups:
        group_edits = edits_by_text.get(group.text_uid, ())
        if not group_edits:
            continue
        group_videos = [videos[video_uid] for video_uid in group.video_uids]
        for start in range(0, len(group_videos), config.evaluation.encode_batch_size):
            current = group_videos[start : start + config.evaluation.encode_batch_size]
            batch, selected = _video_batch(config, current)
            batch = {key: value.to(torch_device) for key, value in batch.items()}
            encoded_video, valid = encode_reference_video(reference, batch)
            for row, (video_record, selected_indexes) in enumerate(
                zip(current, selected, strict=True)
            ):
                q_positives = [edit_vectors[edit.edit_uid][0] for edit in group_edits]
                removal = _feature_removal_positions(
                    encoded_video[row].cpu().numpy(),
                    valid[row].cpu().numpy(),
                    q_positives,
                    selected_indexes,
                    tau=config.auxiliary.tau_support,
                    seed_parts=(config.seed, split, video_record.video_uid),
                )
                corrupted_video = corrupted_valid = None
                if removal["informative"]:
                    corrupted_video, corrupted_valid = _corrupt_and_reencode(
                        reference,
                        batch["video_features"][row : row + 1],
                        batch["video_ignore_raw"][row : row + 1],
                        (removal["selected_positions"], removal["random_positions"]),
                    )
                for edit in group_edits:
                    q_positive, q_negative = edit_vectors[edit.edit_uid]
                    metrics = _support_metrics(
                        encoded_video[row].cpu().numpy(),
                        valid[row].cpu().numpy(),
                        selected_indexes,
                        q_positive,
                        q_negative,
                        tau=config.auxiliary.tau_support,
                        margin=config.auxiliary.margin,
                    )
                    removal_metrics: dict[str, Any] = {
                        "feature_removal_informative": removal["informative"],
                        "feature_removal_count": removal["removal_count"],
                        "selected_removed_feature_rows": list(
                            removal["selected_feature_rows"]
                        ),
                        "random_removed_feature_rows": list(removal["random_feature_rows"]),
                        "feature_removal_corruption": (
                            "zero_mixed_feature_rows_and_mark_ignored_before_reference_encoder"
                            if removal["informative"]
                            else "not_run_fewer_than_four_valid_feature_rows"
                        ),
                    }
                    if corrupted_video is not None and corrupted_valid is not None:
                        selected_metrics = _support_metrics(
                            corrupted_video[0].cpu().numpy(),
                            corrupted_valid[0].cpu().numpy(),
                            selected_indexes,
                            q_positive,
                            q_negative,
                            tau=config.auxiliary.tau_support,
                            margin=config.auxiliary.margin,
                        )
                        random_metrics = _support_metrics(
                            corrupted_video[1].cpu().numpy(),
                            corrupted_valid[1].cpu().numpy(),
                            selected_indexes,
                            q_positive,
                            q_negative,
                            tau=config.auxiliary.tau_support,
                            margin=config.auxiliary.margin,
                        )
                        removal_metrics.update(
                            {
                                "selected_removal_shared_margin": selected_metrics[
                                    "shared_margin"
                                ],
                                "random_removal_shared_margin": random_metrics[
                                    "shared_margin"
                                ],
                                "selected_removal_margin_change": selected_metrics[
                                    "shared_margin"
                                ]
                                - metrics["shared_margin"],
                                "random_removal_margin_change": random_metrics[
                                    "shared_margin"
                                ]
                                - metrics["shared_margin"],
                            }
                        )
                    per_example.append(
                        {
                            "schema_version": 1,
                            "split": split,
                            "diagnostic_only": True,
                            "semantic_validity": "unknown",
                            "video_uid": video_record.video_uid,
                            "group_uid": group.group_uid,
                            "text_uid": group.text_uid,
                            "edit_uid": edit.edit_uid,
                            "occurrence_uid": edit.occurrence_uid,
                            "source_word": edit.source_word,
                            "replacement_word": edit.replacement_word,
                            "prototype_cosine": edit.prototype_cosine,
                            "full_bpe_count": len(tokenized[group.text_uid].full_bpe_ids),
                            "caption_fully_retained": tokenized[
                                group.text_uid
                            ].fully_retained,
                            "caption_duplicate_group_count": split_caption_counts[
                                texts[group.text_uid].caption_hash
                            ],
                            "video_feature_row_count": video_record.num_feature_rows,
                            "video_signer_id": video_record.signer_id,
                            **metrics,
                            **removal_metrics,
                        }
                    )
    fields = (
        "q_difference_norm",
        "text_vector_cosine",
        "support_overlap",
        "support_jsd",
        "positive_support_entropy",
        "positive_support_concentration",
        "peak_displacement_feature_rows",
        "shared_margin",
        "independent_margin",
        "prototype_cosine",
    )
    del reference, model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    gradient_probe = _gradient_probe(
        config,
        upret_root=upret_root,
        device=torch_device,
        tokenizer=tokenizer,
        auxiliary_cache=auxiliary_cache,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "kind": "support",
        "split": split,
        "diagnostic_only": True,
        "fit_split": "train",
        "observation_split": split,
        "semantic_accuracy_computed": False,
        "diagnostic_checkpoint_sha256": sha256_file(config.reference.checkpoint),
        "diagnostic_checkpoint_stage": "initial_selected_baseline",
        "reference_identity_sha256": identity.digest,
        "candidate_graph_sha256": sha256_file(
            cache_root / "auxiliary" / "mining" / "candidate_graph.json"
        ),
        "caption_count": len(groups),
        "caption_with_edit_count": sum(bool(value) for value in edits_by_text.values()),
        "caption_without_edit_count": len(groups)
        - sum(bool(value) for value in edits_by_text.values()),
        "caption_with_edit_fraction": float(
            np.mean([bool(edits_by_text.get(group.text_uid)) for group in groups])
        ),
        "invalid_edit_coverage_definition": (
            "dev captions with no valid alternative from the frozen train-fitted graph"
        ),
        "candidate_edit_count": len(edits),
        "video_edit_observation_count": len(per_example),
        "feature_removal_informative_count": sum(
            item["feature_removal_informative"] for item in per_example
        ),
        "feature_removal_corruption": (
            "zero_mixed_feature_rows_and_mark_ignored_before_reference_encoder"
        ),
        "reliability_gate": {
            "enabled": config.auxiliary.reliability_gate,
            "distribution": (
                "not_computed_by_support_only_diagnostic"
                if config.auxiliary.reliability_gate
                else "not_applicable_for_ungated_initial_diagnostic"
            ),
            "gated_independent_explains_gain": "not_evaluated_without_trained_paired_runs",
        },
        "construction_rejections": mined.rejection_counts,
        "loss_gradient_probe": gradient_probe,
        "shared_active_fraction": float(
            np.mean([item["shared_hinge_active"] for item in per_example])
        ),
        "independent_active_fraction": float(
            np.mean([item["independent_hinge_active"] for item in per_example])
        ),
        "distributions": {
            field: _distribution([float(item[field]) for item in per_example])
            for field in fields
        },
        "feature_removal_distributions": {
            field: _distribution(
                [
                    float(item[field])
                    for item in per_example
                    if item["feature_removal_informative"]
                ]
            )
            for field in (
                "selected_removal_shared_margin",
                "random_removal_shared_margin",
                "selected_removal_margin_change",
                "random_removal_margin_change",
            )
        },
    }
    output_root = Path(config.output.root) / "diagnostics"
    records_path = output_root / f"{split}_support_records.jsonl"
    summary_path = output_root / "diagnostics.json"
    atomic_jsonl_dump(per_example, records_path)
    atomic_json_dump(summary, summary_path)
    summary["records_path"] = str(records_path.resolve())
    summary["summary_path"] = str(summary_path.resolve())
    return summary
