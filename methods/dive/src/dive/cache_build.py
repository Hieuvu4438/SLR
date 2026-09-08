from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    _validate_batch_lineage,
    _verify_input_hashes,
)
from .cache import CacheFingerprint, TensorCacheWriter, TensorShard, make_cache_fingerprint
from .config import config_hash
from .data.manifest import SampleRecord, load_manifest, manifest_hash
from .data.temporal import CompactFrameMap, load_compact_frame_maps
from .data.text_units import load_text_unit_lineage
from .evidence_warmup import _seed_correctness, _text_models, _validate_native_tokens
from .models.evidence import EvidenceEncoder, state_hash


class CacheBuildError(ValueError):
    """Frozen feature cache construction violated stage or provenance contracts."""


_CACHE_NAMES = (
    "native_video",
    "native_text",
    "rgb_local",
    "text_units",
    "reference_local",
)


def _resolve_parents(
    resolver: ArtifactResolver,
) -> tuple[
    ResolvedArtifact,
    ResolvedArtifact,
    ResolvedArtifact,
    ResolvedArtifact,
    ResolvedArtifact,
    ResolvedArtifact,
]:
    try:
        return (
            resolver.resolve("validate_data", "audit", scope="shared"),
            resolver.resolve("validate_data", "native_frame_maps_dir", scope="shared"),
            resolver.resolve("validate_data", "text_unit_maps_dir", scope="shared"),
            resolver.resolve("baseline_validate", "report", scope="shared"),
            resolver.resolve("baseline_train", "locked_checkpoint", scope="shared"),
            resolver.resolve("evidence_warmup", "reference", scope="shared"),
        )
    except ArtifactError as exc:
        raise CacheBuildError(
            "MISSING_PARENT_ARTIFACT: controlled data lineage, baseline, and evidence reference "
            "must be registered before frozen_train cache construction"
        ) from exc


def load_frozen_reference(
    path: str | Path,
    adapter: SedsAdapter,
    *,
    rgb_dim: int,
    pose_dim: int,
    hidden_dim: int,
    output_dim: int,
    normalize_epsilon: float,
    expected_config_hash: str,
    expected_fingerprints: Mapping[str, str],
    device: torch.device,
) -> tuple[EvidenceEncoder, Mapping[str, Any]]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CacheBuildError(f"evidence reference cannot be loaded safely: {source}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "dive_reference.v1":
        raise CacheBuildError("evidence reference schema is invalid")
    if payload.get("config_hash") != expected_config_hash:
        raise CacheBuildError("evidence reference config hash differs from the cache run")
    stored_fingerprints = payload.get("fingerprints")
    if not isinstance(stored_fingerprints, Mapping) or any(
        stored_fingerprints.get(name) != value for name, value in expected_fingerprints.items()
    ):
        raise CacheBuildError("evidence reference parent fingerprints differ from the cache run")
    if (
        payload.get("dtype") != "float32"
        or payload.get("selection_split") != "dev"
        or payload.get("selection_metric") != "mean_t2v_v2t_r1"
    ):
        raise CacheBuildError("evidence reference selection/dtype contract is invalid")
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise CacheBuildError("evidence reference has no model state")
    model = EvidenceEncoder(
        adapter.clone_local_pose_encoder(),
        rgb_dim=rgb_dim,
        pose_dim=pose_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        normalize_epsilon=normalize_epsilon,
    )
    try:
        model.load_state_dict(model_state, strict=True)
    except Exception as exc:
        raise CacheBuildError(
            "evidence reference state is incompatible with its architecture"
        ) from exc
    model.to(device).eval().requires_grad_(False)
    actual_hash = state_hash(model)
    if payload.get("state_hash") != actual_hash:
        raise CacheBuildError("evidence reference state hash mismatch")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in model.state_dict().values()
    ):
        raise CacheBuildError("correctness reference state must remain float32")
    return model, payload


def _fingerprints(
    *,
    config: Mapping[str, Any],
    adapter: SedsAdapter,
    builder: SedsManifestInputBuilder,
    manifest_sha256: str,
    audit_sha256: str,
    native_maps_sha256: str,
    unit_maps_sha256: str,
    baseline_sha256: str,
    reference_sha256: str,
    reference_state_hash: str,
    git_revision: str,
) -> dict[str, CacheFingerprint]:
    preprocessing = dict(adapter.describe_preprocessing())
    temporal = config["temporal"]
    text = config["text"]
    evidence = config["evidence"]
    grid = {
        "view": "canonical",
        "clip_steps": int(temporal["clip_steps"]),
        "stride_steps": int(temporal["dense_stride_steps"]),
        "max_clips": int(temporal["max_clips"]),
        "native_maps": native_maps_sha256,
    }
    native_components = {
        "baseline_state": baseline_sha256,
        "preprocessing": preprocessing,
        "tokenizer": hashlib.sha256(builder.tokenizer_path.read_bytes()).hexdigest(),
        "sequence_selection": {
            "manifest": manifest_sha256,
            "native_frame_maps": native_maps_sha256,
        },
        "masks": {"video": "0_valid_one_cls", "text": "1_valid_through_eot"},
        "sample_ids": {"namespace": "train", "manifest": manifest_sha256},
        "dtype": "float32",
        "implementation_git_revision": git_revision,
    }
    return {
        "native_video": make_cache_fingerprint(
            "native_baseline", native_components | {"payload": "native_video"}
        ),
        "native_text": make_cache_fingerprint(
            "native_baseline", native_components | {"payload": "native_text"}
        ),
        "rgb_local": make_cache_fingerprint(
            "rgb_local",
            {
                "dataset_content": {
                    "manifest": manifest_sha256,
                    "validated_inputs": audit_sha256,
                },
                "extractor_state": {
                    "distribution": "official_seds_released_i3d_features",
                    "content_bound_by_native_audit": native_maps_sha256,
                },
                "frame_map": native_maps_sha256,
                "grid_view": grid,
                "preprocess": preprocessing["rgb_tap"],
                "stage": "frozen_train",
                "dtype": "float32",
                "implementation_git_revision": git_revision,
            },
        ),
        "text_units": make_cache_fingerprint(
            "text_units",
            {
                "text_encoder_state": baseline_sha256,
                "text_model": manifest_sha256,
                "token_ids": unit_maps_sha256,
                "unit_mapping": unit_maps_sha256,
                "normalization": text["normalization_version"],
                "epsilon": float(evidence["normalize_epsilon"]),
                "dtype": "float32",
                "implementation_git_revision": git_revision,
            },
        ),
        "reference_local": make_cache_fingerprint(
            "reference_local",
            {
                "reference_state": {
                    "artifact": reference_sha256,
                    "model": reference_state_hash,
                },
                "rgb_preprocess": preprocessing["rgb_tap"],
                "pose_preprocess": preprocessing["pose_tap"],
                "grid_view": grid,
                "bn_policy": "eval_frozen_running_statistics_and_affine",
                "dtype": "float32",
                "implementation_git_revision": git_revision,
            },
        ),
    }


def _rf_timestamps(
    adapter: SedsAdapter,
    native: Any,
    validity: Tensor,
) -> Tensor:
    output = torch.zeros((*validity.shape, 2), dtype=torch.float32, device=validity.device)
    row_by_id = {sample_id: row for row, sample_id in enumerate(native.sample_ids)}
    observed = torch.zeros_like(validity)
    for record in adapter.describe_receptive_field(native, native.grid_id):
        sample_id = str(record["sample_id"])
        clip = int(record["clip_index"])
        if sample_id not in row_by_id or clip < 0 or clip >= validity.shape[1]:
            raise CacheBuildError("reference RF record addresses an unknown cache position")
        interval = record.get("raw_seconds_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            raise CacheBuildError("reference RF record lacks raw-second provenance")
        row = row_by_id[sample_id]
        output[row, clip] = torch.tensor(interval, dtype=torch.float32, device=output.device)
        observed[row, clip] = True
    if not torch.equal(observed, validity):
        raise CacheBuildError("reference RF records do not exactly cover valid local clips")
    return output


def _expand_rows(values: Tensor, indices: Sequence[int]) -> Tensor:
    return values[torch.as_tensor(indices, dtype=torch.long, device=values.device)]


def _registered_result(
    resolver: ArtifactResolver,
    destination: Path,
    configuration_sha: str,
    expected_parents: Mapping[str, str],
    expected_git_revision: str,
) -> tuple[dict[str, Any], bool] | None:
    registered = True
    try:
        report_parent = resolver.resolve("cache_frozen_train", "report", scope="shared")
    except ArtifactError as exc:
        if "MISSING_PARENT_ARTIFACT" not in str(exc):
            raise CacheBuildError(str(exc)) from exc
        if not destination.exists():
            return None
        report_parent = None
        registered = False
    if registered:
        for name in _CACHE_NAMES:
            resolver.resolve("cache_frozen_train", name, scope="shared")
        assert report_parent is not None
        report_path = report_parent.path
    else:
        report_path = destination / "report.json"
        if any(not (destination / name / "index.json").is_file() for name in _CACHE_NAMES):
            raise CacheBuildError(
                f"unregistered frozen cache destination is incomplete: {destination}"
            )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheBuildError("registered frozen cache report is unreadable") from exc
    if (
        report.get("schema_version") != "dive_frozen_train_cache.v1"
        or report.get("config_sha256") != configuration_sha
        or report.get("parents") != dict(expected_parents)
        or report.get("implementation_git_revision") != expected_git_revision
    ):
        raise CacheBuildError("CACHE_HASH_MISMATCH: registered frozen cache provenance differs")
    if not registered:
        stored_fingerprints = report.get("fingerprints")
        if not isinstance(stored_fingerprints, Mapping):
            raise CacheBuildError("unregistered frozen cache has no fingerprint records")
        for name in _CACHE_NAMES:
            raw = stored_fingerprints.get(name)
            if not isinstance(raw, Mapping):
                raise CacheBuildError(f"unregistered {name} cache fingerprint is missing")
            try:
                fingerprint = make_cache_fingerprint(str(raw["artifact"]), raw["components"])
            except (KeyError, TypeError) as exc:
                raise CacheBuildError(f"unregistered {name} cache fingerprint is invalid") from exc
            if fingerprint.to_dict() != dict(raw):
                raise CacheBuildError(
                    f"CACHE_HASH_MISMATCH: unregistered {name} fingerprint differs"
                )
            TensorCacheWriter(
                destination / name,
                namespace=f"train_{name}",
                fingerprint=fingerprint,
                resume=True,
            ).finalize()
    return report, registered


def _register_cache_result(
    resolver: ArtifactResolver,
    destination: Path,
    report: Mapping[str, Any],
    parents: Sequence[ResolvedArtifact],
) -> dict[str, Any]:
    registered = resolver.record_stage(
        "cache_frozen_train",
        {
            **{name: destination / name for name in _CACHE_NAMES},
            "report": destination / "report.json",
        },
        scope="shared",
        parents=parents,
        metadata={
            "kind": "frozen_train",
            "split": "train",
            "record_count": int(report["manifest"]["record_count"]),
            "batch_size": int(report["batch_size"]),
        },
    )
    result = dict(report)
    result["artifacts"] = {
        name: {"artifact_id": item.artifact_id, "sha256": item.sha256}
        for name, item in registered.items()
    }
    return result


def build_frozen_train_cache(
    config: Mapping[str, Any],
    *,
    batch_size: int = 32,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Build all frozen train features once with sample-addressed, checksummed shards."""
    if batch_size <= 0:
        raise CacheBuildError("cache batch size must be positive")
    data = _mapping(config, "data")
    baseline = _mapping(config, "baseline")
    evidence = _mapping(config, "evidence")
    if baseline.get("family") != "seds":
        raise CacheBuildError("frozen_train cache currently requires family=seds")
    resolver = ArtifactResolver(config)
    configuration_sha = config_hash(config)
    destination = resolver.output_path("shared", "cache", "frozen_train")
    audit, native_maps_parent, units_parent, baseline_report, checkpoint, reference = (
        _resolve_parents(resolver)
    )
    parent_hashes = {
        "data_audit": audit.record.sha256,
        "native_frame_maps": native_maps_parent.record.sha256,
        "text_unit_maps": units_parent.record.sha256,
        "baseline_validation": baseline_report.record.sha256,
        "baseline_checkpoint": checkpoint.record.sha256,
        "reference": reference.record.sha256,
    }
    revision = _repository_revision(Path(__file__).resolve().parents[4])
    existing = _registered_result(
        resolver,
        destination,
        configuration_sha,
        parent_hashes,
        revision,
    )
    if existing is not None:
        report, registered = existing
        if registered:
            return report
        return _register_cache_result(
            resolver,
            destination,
            report,
            (audit, native_maps_parent, units_parent, baseline_report, checkpoint, reference),
        )
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise CacheBuildError("native SEDS frozen cache construction requires CUDA")
    _seed_correctness(int(config["run"]["seed"]))

    manifest_path = _configured_or_prepared(data, resolver, "train_manifest", "train_manifest")
    frame_maps_dir = _configured_or_prepared(data, resolver, "frame_maps_dir", "frame_maps_dir")
    upstream_root = _required_path(data.get("upstream_root"), "data.upstream_root")
    pose_root = _required_path(data.get("pose_root"), "data.pose_root")
    rgb_root = _required_path(data.get("rgb_cache_root"), "data.rgb_cache_root")
    reproduction_path = _required_path(
        baseline.get("reproduction_config"), "baseline.reproduction_config"
    )
    reproduction = load_seds_reproduction(reproduction_path, upstream_root=upstream_root)
    records = load_manifest(manifest_path, expected_split="train")
    frame_maps_sequence = load_compact_frame_maps(
        frame_maps_dir / "train.jsonl",
        expected_sample_ids=[item.sample_id for item in records],
    )
    frame_maps: dict[str, CompactFrameMap] = {item.sample_id: item for item in frame_maps_sequence}
    native_rows = _load_native_frame_rows(native_maps_parent.path / "train.jsonl", records)
    _verify_input_hashes(records, native_rows, pose_root=pose_root, rgb_root=rgb_root)
    lineage = load_text_unit_lineage(
        units_parent.path / "train.jsonl", expected_texts=_text_models(records)
    )

    adapter_config = copy.deepcopy(dict(config))
    adapter_config["baseline"] = dict(baseline)
    adapter_config["baseline"]["locked_checkpoint"] = str(checkpoint.path)
    adapter = SedsAdapter.from_official_checkpoint(
        checkpoint.path,
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
    reference_model, reference_payload = load_frozen_reference(
        reference.path,
        adapter,
        rgb_dim=int(reproduction.model_arguments["rgb_dim"]),
        pose_dim=int(reproduction.model_arguments["pose_dim"]),
        hidden_dim=int(evidence["hidden_dim"]),
        output_dim=int(evidence["output_dim"]),
        normalize_epsilon=float(evidence["normalize_epsilon"]),
        expected_config_hash=config_hash(config),
        expected_fingerprints={
            "baseline": checkpoint.record.sha256,
            "data": audit.record.sha256,
            "grid": native_maps_parent.record.sha256,
            "units": units_parent.record.sha256,
        },
        device=target_device,
    )
    manifest_sha256 = manifest_hash(manifest_path)
    fingerprints = _fingerprints(
        config=config,
        adapter=adapter,
        builder=builder,
        manifest_sha256=manifest_sha256,
        audit_sha256=audit.record.sha256,
        native_maps_sha256=native_maps_parent.record.sha256,
        unit_maps_sha256=units_parent.record.sha256,
        baseline_sha256=checkpoint.record.sha256,
        reference_sha256=reference.record.sha256,
        reference_state_hash=str(reference_payload["state_hash"]),
        git_revision=revision,
    )

    if destination.exists():  # guarded before expensive model construction
        raise CacheBuildError(f"unregistered frozen cache destination exists: {destination}")
    else:
        pending = destination.with_name(f"{destination.name}.pending")
        build_state = {
            "schema_version": "dive_frozen_train_cache_build.v1",
            "config_sha256": configuration_sha,
            "implementation_git_revision": revision,
            "batch_size": batch_size,
            "manifest_sha256": manifest_sha256,
            "record_count": len(records),
            "parents": parent_hashes,
            "fingerprints": {name: value.to_dict() for name, value in fingerprints.items()},
        }
        if pending.exists():
            state_path = pending / "build_state.json"
            try:
                stored_state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CacheBuildError("pending frozen cache build state is unreadable") from exc
            if stored_state != build_state:
                raise CacheBuildError(
                    "CACHE_HASH_MISMATCH: pending frozen cache build provenance differs"
                )
        else:
            pending.mkdir(parents=True, exist_ok=False)
            _atomic_json(pending / "build_state.json", build_state)
        writers = {
            name: TensorCacheWriter(
                pending / name,
                namespace=f"train_{name}",
                fingerprint=fingerprints[name],
                resume=True,
            )
            for name in _CACHE_NAMES
        }
        expected_shards: dict[str, tuple[str, ...]] = {}
        with torch.no_grad():
            for shard_index, start in enumerate(range(0, len(records), batch_size)):
                part = records[start : start + batch_size]
                shard_id = f"{shard_index:05d}"
                sample_ids = tuple(item.sample_id for item in part)
                expected_shards[shard_id] = sample_ids
                present = {
                    name: shard_id in writer.completed_shard_ids for name, writer in writers.items()
                }
                for name, is_present in present.items():
                    if is_present and writers[name].ordered_ids_for(shard_id) != sample_ids:
                        raise CacheBuildError(
                            f"pending {name} shard {shard_id} IDs differ from the manifest"
                        )
                if all(present.values()):
                    continue
                native = builder.build_video_batch(
                    part,
                    raw_frame_counts=[
                        frame_maps[item.sample_id].video_frame_count for item in part
                    ],
                    frames_per_second=[frame_maps[item.sample_id].fps for item in part],
                )
                _validate_batch_lineage(native, part, native_rows)
                native = _video_to(native, target_device)
                native_features = adapter.encode_video_native(native)
                rgb = adapter.rgb_local_features(native, native.grid_id)
                grid = adapter.local_pose_grid(native, native.grid_id)
                pose = {
                    "right": native.right_pose,
                    "left": native.left_pose,
                    "body": native.body_pose,
                }
                reference_features = reference_model(
                    pose, rgb.streams["rgb_local"], grid, rgb.validity
                )
                timestamps = _rf_timestamps(adapter, native, rgb.validity)
                common_metadata = {
                    "split": "train",
                    "manifest_offset": start,
                    "grid_id": native.grid_id,
                }
                native_video_shard = TensorShard(
                    shard_id=shard_id,
                    ordered_ids=sample_ids,
                    tensors={
                        "pooled": native_features.pooled.float(),
                        "fusion_hidden": native_features.streams["fusion_hidden"].float(),
                    },
                    masks={"video": native_features.validity},
                    timestamps={},
                    metadata=common_metadata | {"contains_cls": True},
                )
                rgb_local_shard = TensorShard(
                    shard_id=shard_id,
                    ordered_ids=sample_ids,
                    tensors={"rgb_local": rgb.streams["rgb_local"].float(), "grid": grid},
                    masks={"video": rgb.validity},
                    timestamps={"raw_seconds": timestamps},
                    metadata=common_metadata | {"contains_cls": False},
                )
                reference_local_shard = TensorShard(
                    shard_id=shard_id,
                    ordered_ids=sample_ids,
                    tensors={"reference": reference_features.float(), "grid": grid},
                    masks={"video": rgb.validity},
                    timestamps={"raw_seconds": timestamps},
                    metadata=common_metadata | {"contains_cls": False},
                )

                unique: dict[str, SampleRecord] = {}
                for record in part:
                    unique.setdefault(record.text_id, record)
                text_records = tuple(unique.values())
                native_text_batch = builder.build_text_batch(text_records)
                _validate_native_tokens(
                    native_text_batch.text_ids, native_text_batch.input_ids, lineage
                )
                native_text = adapter.encode_text_native(_text_to(native_text_batch, target_device))
                unit_text = adapter.pool_text_units(
                    native_text,
                    [lineage[text_id].units for text_id in native_text.text_ids],
                )
                row_by_text = {text_id: row for row, text_id in enumerate(native_text.text_ids)}
                row_indices = [row_by_text[item.text_id] for item in part]
                text_ids = [item.text_id for item in part]
                native_text_shard = TensorShard(
                    shard_id=shard_id,
                    ordered_ids=sample_ids,
                    tensors={
                        "pooled": _expand_rows(native_text.pooled.float(), row_indices),
                        "tokens": _expand_rows(native_text.token_features.float(), row_indices),
                    },
                    masks={"text": _expand_rows(native_text.token_validity, row_indices)},
                    timestamps={},
                    metadata=common_metadata | {"text_ids": text_ids},
                )
                text_units_shard = TensorShard(
                    shard_id=shard_id,
                    ordered_ids=sample_ids,
                    tensors={"units": _expand_rows(unit_text.token_features.float(), row_indices)},
                    masks={"units": _expand_rows(unit_text.token_validity, row_indices)},
                    timestamps={},
                    metadata=common_metadata | {"text_ids": text_ids},
                )
                shard_values = {
                    "native_video": native_video_shard,
                    "native_text": native_text_shard,
                    "rgb_local": rgb_local_shard,
                    "text_units": text_units_shard,
                    "reference_local": reference_local_shard,
                }
                for name, shard in shard_values.items():
                    if not present[name]:
                        writers[name].add(shard)
        expected_ids = frozenset(expected_shards)
        for name, writer in writers.items():
            if writer.completed_shard_ids != expected_ids:
                raise CacheBuildError(f"{name} cache does not cover every planned shard")
        for writer in writers.values():
            writer.finalize()
        report = {
            "schema_version": "dive_frozen_train_cache.v1",
            "ready": True,
            "kind": "frozen_train",
            "config_sha256": config_hash(config),
            "implementation_git_revision": revision,
            "test_content_used": False,
            "manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "record_count": len(records),
            },
            "batch_size": batch_size,
            "shard_count": (len(records) + batch_size - 1) // batch_size,
            "fingerprints": {name: value.to_dict() for name, value in fingerprints.items()},
            "cache_paths": {name: str(destination / name) for name in _CACHE_NAMES},
            "parents": parent_hashes,
        }
        _atomic_json(pending / "report.json", report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending, destination)
    return _register_cache_result(
        resolver,
        destination,
        report,
        (audit, native_maps_parent, units_parent, baseline_report, checkpoint, reference),
    )
