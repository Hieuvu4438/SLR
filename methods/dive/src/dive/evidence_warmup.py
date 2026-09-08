from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .adapters import SedsAdapter, SedsManifestInputBuilder, load_seds_reproduction
from .artifacts import ArtifactError, ArtifactResolver, ResolvedArtifact
from .baseline import (
    _atomic_json,
    _configured_or_prepared,
    _load_native_frame_rows,
    _mapping,
    _required_path,
    _text_to,
    _video_to,
)
from .baseline_training import (
    _repository_revision,
    _unique_text_records,
    _validate_batch_lineage,
    _verify_input_hashes,
    build_baseline_epoch_plan,
)
from .config import config_hash
from .data.manifest import SampleRecord, load_manifest, manifest_hash
from .data.relations import build_pair_relations, load_excluded_negatives, relations_hash
from .data.relevance import load_relevance, relevance_hash
from .data.temporal import CompactFrameMap, load_compact_frame_maps
from .data.text_units import TextUnitLineage, load_text_unit_lineage
from .models.evidence import EvidenceEncoder
from .training.warmup import (
    ChunkedWarmupGallery,
    WarmupBatch,
    WarmupTextGalleryBatch,
    WarmupVideoGalleryBatch,
    run_evidence_warmup,
)


class EvidenceWarmupError(ValueError):
    """The native evidence warm-up cannot satisfy its data or provenance contract."""


def _resolve_parents(
    resolver: ArtifactResolver,
) -> tuple[
    ResolvedArtifact, ResolvedArtifact, ResolvedArtifact, ResolvedArtifact, ResolvedArtifact
]:
    try:
        return (
            resolver.resolve("validate_data", "audit", scope="shared"),
            resolver.resolve("validate_data", "native_frame_maps_dir", scope="shared"),
            resolver.resolve("validate_data", "text_unit_maps_dir", scope="shared"),
            resolver.resolve("baseline_validate", "report", scope="shared"),
            resolver.resolve("baseline_train", "locked_checkpoint", scope="shared"),
        )
    except ArtifactError as exc:
        raise EvidenceWarmupError(
            "MISSING_PARENT_ARTIFACT: validate_data audit/frame/unit maps, baseline validation, "
            "and the controlled locked checkpoint must pass before evidence warm-up"
        ) from exc


def _text_models(records: Sequence[SampleRecord]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        previous = result.setdefault(record.text_id, record.text_model)
        if previous != record.text_model:
            raise EvidenceWarmupError(f"inconsistent text for text ID: {record.text_id}")
    return result


def _validate_native_tokens(
    text_ids: Sequence[str], token_ids: Tensor, lineage: Mapping[str, TextUnitLineage]
) -> None:
    if len(text_ids) != len(token_ids):
        raise EvidenceWarmupError("native token rows do not match text IDs")
    for row, text_id in enumerate(text_ids):
        expected = lineage[text_id].token_ids
        if tuple(int(value) for value in token_ids[row].tolist()) != expected:
            raise EvidenceWarmupError(
                f"native SEDS token IDs changed after text-unit validation: {text_id}"
            )


def _encode_text_cache(
    adapter: SedsAdapter,
    builder: SedsManifestInputBuilder,
    records: Sequence[SampleRecord],
    lineage: Mapping[str, TextUnitLineage],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Tensor]:
    cache: dict[str, Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            part = records[start : start + batch_size]
            native = builder.build_text_batch(part)
            _validate_native_tokens(native.text_ids, native.input_ids, lineage)
            features = adapter.encode_text_units(
                _text_to(native, device),
                [lineage[text_id].units for text_id in native.text_ids],
            )
            for row, text_id in enumerate(features.text_ids):
                valid = features.token_validity[row]
                if not bool(valid.any()):
                    raise EvidenceWarmupError(f"text has no complete local units: {text_id}")
                value = features.token_features[row, valid].detach().float().cpu().contiguous()
                if text_id in cache:
                    raise EvidenceWarmupError(f"duplicate cached text ID: {text_id}")
                cache[text_id] = value
    if set(cache) != set(lineage):
        raise EvidenceWarmupError("frozen text cache does not exactly cover validated unit maps")
    return cache


def collate_cached_text_units(
    text_ids: Sequence[str], cache: Mapping[str, Tensor], *, device: torch.device
) -> tuple[Tensor, Tensor]:
    if not text_ids or len(text_ids) != len(set(text_ids)):
        raise EvidenceWarmupError("text feature batch requires unique nonempty IDs")
    missing = [text_id for text_id in text_ids if text_id not in cache]
    if missing:
        raise EvidenceWarmupError(f"text feature cache is missing IDs: {missing}")
    values = [cache[text_id] for text_id in text_ids]
    dimensions = {tuple(value.shape[1:]) for value in values}
    if len(dimensions) != 1 or any(value.ndim != 2 or value.shape[0] <= 0 for value in values):
        raise EvidenceWarmupError("cached text-unit tensors have inconsistent shapes")
    maximum = max(len(value) for value in values)
    output = values[0].new_zeros((len(values), maximum, values[0].shape[-1]))
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for row, value in enumerate(values):
        output[row, : len(value)] = value
        mask[row, : len(value)] = True
    return output.to(device), mask.to(device)


def _video_components(
    adapter: SedsAdapter,
    builder: SedsManifestInputBuilder,
    records: Sequence[SampleRecord],
    frame_maps: Mapping[str, CompactFrameMap],
    native_rows: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[Mapping[str, Tensor], Tensor, Tensor, Tensor]:
    native = builder.build_video_batch(
        records,
        raw_frame_counts=[frame_maps[item.sample_id].video_frame_count for item in records],
        frames_per_second=[frame_maps[item.sample_id].fps for item in records],
    )
    _validate_batch_lineage(native, records, native_rows)
    native = _video_to(native, device)
    rgb = adapter.rgb_local_features(native, native.grid_id)
    pose = {
        "right": native.right_pose,
        "left": native.left_pose,
        "body": native.body_pose,
    }
    if rgb.streams.get("rgb_local") is None:
        raise EvidenceWarmupError("SEDS adapter omitted the pre-context local RGB tap")
    grid = adapter.local_pose_grid(native, native.grid_id)
    return pose, rgb.streams["rgb_local"], grid, rgb.validity


def _seed_correctness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def warmup_seds_evidence(
    config: Mapping[str, Any],
    *,
    batch_size: int = 128,
    eval_video_batch_size: int | None = None,
    eval_text_batch_size: int | None = None,
    device: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    """Warm up DIVE local evidence on ordinary pairs and register a frozen dev winner."""
    data = _mapping(config, "data")
    baseline = _mapping(config, "baseline")
    evidence = _mapping(config, "evidence")
    sampler = _mapping(config, "sampler")
    evaluation = _mapping(config, "evaluation")
    if baseline.get("family") != "seds":
        raise EvidenceWarmupError("native evidence warm-up currently requires family=seds")
    expected_batch_size = int(sampler["effective_batch_size"])
    if batch_size != expected_batch_size:
        raise EvidenceWarmupError(f"warm-up batch size must be {expected_batch_size}")
    video_batch_size = int(
        evaluation["query_chunk"] if eval_video_batch_size is None else eval_video_batch_size
    )
    text_batch_size = int(
        evaluation["candidate_chunk"] if eval_text_batch_size is None else eval_text_batch_size
    )
    if video_batch_size <= 0 or text_batch_size <= 0:
        raise EvidenceWarmupError("warm-up evaluation batch sizes must be positive")
    if (
        int(config["train"]["world_size"]) != 1
        or int(config["train"]["gradient_accumulation_steps"]) != 1
    ):
        raise EvidenceWarmupError("correctness warm-up requires world_size=accumulation=1")
    resolver = ArtifactResolver(config)
    audit_parent, native_parent, units_parent, baseline_report, checkpoint_parent = (
        _resolve_parents(resolver)
    )
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise EvidenceWarmupError("native SEDS evidence warm-up requires CUDA")
    train_manifest = _configured_or_prepared(data, resolver, "train_manifest", "train_manifest")
    dev_manifest = _configured_or_prepared(data, resolver, "dev_manifest", "dev_manifest")
    relations_path = _configured_or_prepared(data, resolver, "train_relations", "train_relations")
    frame_maps_dir = _configured_or_prepared(data, resolver, "frame_maps_dir", "frame_maps_dir")
    relevance_dir = _configured_or_prepared(data, resolver, "relevance_dir", "relevance_dir")
    upstream_root = _required_path(data.get("upstream_root"), "data.upstream_root")
    pose_root = _required_path(data.get("pose_root"), "data.pose_root")
    rgb_root = _required_path(data.get("rgb_cache_root"), "data.rgb_cache_root")
    reproduction_path = _required_path(
        baseline.get("reproduction_config"), "baseline.reproduction_config"
    )
    reproduction = load_seds_reproduction(reproduction_path, upstream_root=upstream_root)

    train_records = load_manifest(train_manifest, expected_split="train")
    dev_records = load_manifest(dev_manifest, expected_split="dev")
    train_maps_sequence = load_compact_frame_maps(
        frame_maps_dir / "train.jsonl",
        expected_sample_ids=[item.sample_id for item in train_records],
    )
    dev_maps_sequence = load_compact_frame_maps(
        frame_maps_dir / "dev.jsonl",
        expected_sample_ids=[item.sample_id for item in dev_records],
    )
    train_maps = {item.sample_id: item for item in train_maps_sequence}
    dev_maps = {item.sample_id: item for item in dev_maps_sequence}
    train_native_rows = _load_native_frame_rows(native_parent.path / "train.jsonl", train_records)
    dev_native_rows = _load_native_frame_rows(native_parent.path / "dev.jsonl", dev_records)
    _verify_input_hashes(train_records, train_native_rows, pose_root=pose_root, rgb_root=rgb_root)
    _verify_input_hashes(dev_records, dev_native_rows, pose_root=pose_root, rgb_root=rgb_root)

    train_text_records = _unique_text_records(train_records)
    dev_text_records = _unique_text_records(dev_records)
    train_lineage = load_text_unit_lineage(
        units_parent.path / "train.jsonl", expected_texts=_text_models(train_records)
    )
    dev_lineage = load_text_unit_lineage(
        units_parent.path / "dev.jsonl", expected_texts=_text_models(dev_records)
    )
    train_relevance_path = relevance_dir / "train.jsonl"
    dev_relevance_path = relevance_dir / "dev.jsonl"
    train_text_ids = tuple(item.text_id for item in train_text_records)
    dev_text_ids = tuple(item.text_id for item in dev_text_records)
    train_relevance = load_relevance(
        train_relevance_path,
        video_ids=[item.video_id for item in train_records],
        text_ids=train_text_ids,
    )
    dev_relevance = load_relevance(
        dev_relevance_path,
        video_ids=[item.video_id for item in dev_records],
        text_ids=dev_text_ids,
    )
    excluded_records = load_excluded_negatives(
        relations_path,
        video_ids=[item.video_id for item in train_records],
        text_ids=train_text_ids,
        positives_by_video=train_relevance,
    )
    excluded_pairs = frozenset((item.video_id, item.text_id) for item in excluded_records)

    repository_root = Path(__file__).resolve().parents[4]
    revision = _repository_revision(repository_root)
    seed = int(config["run"]["seed"])
    _seed_correctness(seed)
    adapter_config = copy.deepcopy(dict(config))
    adapter_config["baseline"] = dict(baseline)
    adapter_config["baseline"]["locked_checkpoint"] = str(checkpoint_parent.path)
    adapter = SedsAdapter.from_official_checkpoint(
        checkpoint_parent.path,
        adapter_config,
        upstream_root=upstream_root,
        device=target_device,
        temperature=float(evidence["tau_alignment"]),
    )
    builder = SedsManifestInputBuilder(
        upstream_root=upstream_root,
        reproduction_config=reproduction_path,
        pose_root=pose_root,
        rgb_root=rgb_root,
    )
    train_text_cache = _encode_text_cache(
        adapter,
        builder,
        train_text_records,
        train_lineage,
        batch_size=text_batch_size,
        device=target_device,
    )
    dev_text_cache = _encode_text_cache(
        adapter,
        builder,
        dev_text_records,
        dev_lineage,
        batch_size=text_batch_size,
        device=target_device,
    )
    adapter.model.to("cpu")
    model = EvidenceEncoder(
        adapter.clone_local_pose_encoder(),
        rgb_dim=int(reproduction.model_arguments["rgb_dim"]),
        pose_dim=int(reproduction.model_arguments["pose_dim"]),
        hidden_dim=int(evidence["hidden_dim"]),
        output_dim=int(evidence["output_dim"]),
        normalize_epsilon=float(evidence["normalize_epsilon"]),
    ).to(target_device)
    torch.cuda.empty_cache()

    def train_batches(epoch: int):
        plan = build_baseline_epoch_plan(train_records, seed=seed, epoch=epoch)
        for start in range(0, len(plan.records), batch_size):
            part = plan.records[start : start + batch_size]
            pose, rgb, grid, video_mask = _video_components(
                adapter,
                builder,
                part,
                train_maps,
                train_native_rows,
                device=target_device,
            )
            ids = tuple(item.text_id for item in part)
            text_units, text_mask = collate_cached_text_units(
                ids, train_text_cache, device=target_device
            )
            relations = build_pair_relations(
                [item.video_id for item in part],
                ids,
                train_relevance,
                excluded_pairs,
            )
            yield WarmupBatch(
                pose=pose,
                rgb_local=rgb,
                grid=grid,
                video_mask=video_mask,
                text_units=text_units,
                text_mask=text_mask,
                positives=relations.positives.to(target_device),
                candidates=relations.candidates.to(target_device),
            )

    def dev_video_batches():
        for start in range(0, len(dev_records), video_batch_size):
            part = dev_records[start : start + video_batch_size]
            pose, rgb, grid, video_mask = _video_components(
                adapter,
                builder,
                part,
                dev_maps,
                dev_native_rows,
                device=target_device,
            )
            yield WarmupVideoGalleryBatch(
                pose=pose,
                rgb_local=rgb,
                grid=grid,
                video_mask=video_mask,
                video_ids=tuple(item.video_id for item in part),
            )

    def dev_text_batches():
        for start in range(0, len(dev_text_ids), text_batch_size):
            ids = dev_text_ids[start : start + text_batch_size]
            text_units, text_mask = collate_cached_text_units(
                ids, dev_text_cache, device=target_device
            )
            yield WarmupTextGalleryBatch(
                text_units=text_units,
                text_mask=text_mask,
                text_ids=ids,
            )

    gallery = ChunkedWarmupGallery(
        video_batches=dev_video_batches,
        text_batches=dev_text_batches,
        video_ids=tuple(item.video_id for item in dev_records),
        text_ids=dev_text_ids,
        video_to_text_positives=dev_relevance,
    )
    steps_per_epoch = math.ceil(len(train_text_records) / batch_size)
    fingerprints = {
        "baseline": checkpoint_parent.record.sha256,
        "data": audit_parent.record.sha256,
        "grid": native_parent.record.sha256,
        "units": units_parent.record.sha256,
        "baseline_validation": baseline_report.record.sha256,
        "train_manifest": manifest_hash(train_manifest),
        "dev_manifest": manifest_hash(dev_manifest),
        "train_relations": relations_hash(relations_path),
        "train_relevance": relevance_hash(train_relevance_path),
        "dev_relevance": relevance_hash(dev_relevance_path),
        "reproduction": hashlib.sha256(reproduction_path.read_bytes()).hexdigest(),
    }
    output_dir = resolver.output_path("shared", "evidence", "warmup")
    result = run_evidence_warmup(
        model,
        train_batches,
        gallery,
        config=config,
        steps_per_epoch=steps_per_epoch,
        output_dir=output_dir,
        config_hash=config_hash(config),
        fingerprints=fingerprints,
        git_revision=revision,
        forbidden_modules=(adapter.model,),
        resume=resume,
    )
    report_path = output_dir / "report.json"
    report: dict[str, Any] = {
        "schema_version": "dive_evidence_warmup.v1",
        "ready": True,
        "config_sha256": config_hash(config),
        "implementation_git_revision": revision,
        "test_content_used_for_tuning": False,
        "baseline_checkpoint_sha256": checkpoint_parent.record.sha256,
        "fingerprints": fingerprints,
        "train": {
            "manifest": str(train_manifest),
            "record_count": len(train_records),
            "unique_text_count": len(train_text_records),
            "excluded_negative_count": len(excluded_records),
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "epochs": int(config["train"]["warmup_epochs"]),
        },
        "dev": {
            "manifest": str(dev_manifest),
            "video_count": len(dev_records),
            "text_count": len(dev_text_ids),
            "video_batch_size": video_batch_size,
            "text_batch_size": text_batch_size,
            "selection_split": "dev",
        },
        "selection": {
            "winner": asdict(result.winner) | {"endpoint": result.winner.endpoint},
            "candidates": [
                asdict(item) | {"endpoint": item.endpoint} for item in result.candidates
            ],
            "reference_state_hash": result.reference_state_hash,
            "reference_sha256": result.reference_checksum,
        },
        "optimizer": dict(result.optimizer_manifest),
    }
    _atomic_json(report_path, report)
    registered = resolver.record_stage(
        "evidence_warmup",
        {
            "reference": result.reference_path,
            "report": report_path,
            "selection": result.selection_path,
        },
        scope="shared",
        parents=(
            audit_parent,
            native_parent,
            units_parent,
            baseline_report,
            checkpoint_parent,
        ),
        metadata={
            "selection_split": "dev",
            "selection_metric": "mean_t2v_v2t_r1",
            "winner_epoch": result.winner.epoch,
            "optimizer_steps": int(config["train"]["warmup_epochs"]) * steps_per_epoch,
        },
    )
    report["artifacts"] = {
        name: {"artifact_id": item.artifact_id, "sha256": item.sha256}
        for name, item in registered.items()
    }
    return report
