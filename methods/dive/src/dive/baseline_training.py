from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .adapters import (
    NativeTextFeatures,
    NativeVideoFeatures,
    SedsAdapter,
    SedsManifestInputBuilder,
    SedsTrainingBatch,
    hash_seds_input,
    load_seds_reproduction,
)
from .adapters.seds_reproduction import verify_seds_checkout
from .artifacts import ArtifactError, ArtifactResolver, ResolvedArtifact, hash_artifact
from .baseline import (
    _atomic_json,
    _atomic_scores,
    _configured_or_prepared,
    _cpu_text,
    _cpu_video,
    _join_text,
    _join_video,
    _load_native_frame_rows,
    _mapping,
    _required_path,
    _text_to,
    _video_to,
    score_feature_gallery,
)
from .config import config_hash
from .data.manifest import SampleRecord, load_manifest, manifest_hash
from .data.relevance import load_relevance, relevance_hash
from .data.temporal import CompactFrameMap, load_compact_frame_maps
from .eval.metrics import RetrievalMetrics, evaluate_retrieval
from .training.selection import DevCandidate, select_earliest_best
from .training.state import CheckpointError, load_training_checkpoint, save_training_checkpoint


class BaselineTrainingError(ValueError):
    """The controlled native SEDS baseline cannot be trained reproducibly."""


@dataclass(frozen=True)
class BaselineEpochPlan:
    epoch: int
    records: tuple[SampleRecord, ...]
    ordered_sample_ids: tuple[str, ...]
    ordered_text_ids: tuple[str, ...]
    sha256: str
    augmentation_seed: int


@dataclass(frozen=True)
class BaselineStepMetrics:
    loss: float
    loss_fusion: float
    loss_pose: float
    loss_rgb: float
    kl_pose: float
    kl_rgb: float
    loss_rgb_pose: float
    global_grad_norm: float


def _derived_seed(seed: int, epoch: int, purpose: str) -> int:
    payload = f"seds_controlled_v1:{seed}:{epoch}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_baseline_epoch_plan(
    records: Sequence[SampleRecord], *, seed: int, epoch: int
) -> BaselineEpochPlan:
    """Select exactly one video view per text ID and deterministically shuffle the epoch."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or epoch <= 0:
        raise BaselineTrainingError("epoch-plan seed/epoch are invalid")
    groups: dict[str, list[SampleRecord]] = {}
    text_by_id: dict[str, str] = {}
    for record in records:
        if record.split != "train":
            raise BaselineTrainingError("baseline epoch plans accept train records only")
        previous = text_by_id.setdefault(record.text_id, record.text_model)
        if previous != record.text_model:
            raise BaselineTrainingError(f"inconsistent text for text ID: {record.text_id}")
        groups.setdefault(record.text_id, []).append(record)
    if not groups:
        raise BaselineTrainingError("baseline epoch plan cannot be empty")
    generator = random.Random(_derived_seed(seed, epoch, "view_and_order"))
    selected = [
        generator.choice(sorted(groups[text_id], key=lambda item: item.sample_id))
        for text_id in sorted(groups)
    ]
    generator.shuffle(selected)
    sample_ids = tuple(item.sample_id for item in selected)
    text_ids = tuple(item.text_id for item in selected)
    if len(sample_ids) != len(set(sample_ids)) or len(text_ids) != len(set(text_ids)):
        raise BaselineTrainingError("epoch plan violated unique sample/text IDs")
    digest = hashlib.sha256(
        json.dumps(
            list(zip(sample_ids, text_ids, strict=True)),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return BaselineEpochPlan(
        epoch=epoch,
        records=tuple(selected),
        ordered_sample_ids=sample_ids,
        ordered_text_ids=text_ids,
        sha256=digest,
        augmentation_seed=_derived_seed(seed, epoch, "augmentation"),
    )


def _load_pinned_optimizer(upstream_root: Path) -> type[torch.optim.Optimizer]:
    source = upstream_root / "modules" / "optimization.py"
    if source.is_symlink() or not source.is_file():
        raise BaselineTrainingError(f"pinned SEDS optimizer source is missing: {source}")
    module = ModuleType("dive_pinned_seds_optimization")
    module.__file__ = str(source)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    except Exception as exc:
        raise BaselineTrainingError("pinned SEDS optimizer source could not be loaded") from exc
    finally:
        sys.dont_write_bytecode = previous
    optimizer = getattr(module, "BertAdam", None)
    if not isinstance(optimizer, type) or not issubclass(optimizer, torch.optim.Optimizer):
        raise BaselineTrainingError("pinned SEDS optimizer class is invalid")
    return optimizer


def build_seds_optimizer(
    model: nn.Module,
    *,
    upstream_root: str | Path,
    controlled_training: Mapping[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, Mapping[str, Any]]:
    """Reproduce the six published SEDS parameter groups with explicit ownership."""
    root = verify_seds_checkout(upstream_root)
    if total_steps <= 0:
        raise BaselineTrainingError("optimizer total_steps must be positive")
    named = list(model.named_parameters())
    if not named:
        raise BaselineTrainingError("SEDS model has no parameters")
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")

    def branch(name: str) -> str:
        if "clip." in name or "clip_rgb." in name:
            return "clip"
        if "signbert." in name:
            return "signbert"
        return "other"

    buckets: dict[tuple[bool, str], list[tuple[str, nn.Parameter]]] = {
        (decay, family): [] for decay in (True, False) for family in ("clip", "signbert", "other")
    }
    for name, parameter in named:
        buckets[(not any(value in name for value in no_decay), branch(name))].append(
            (name, parameter)
        )
    order = (
        (True, "clip"),
        (True, "signbert"),
        (True, "other"),
        (False, "clip"),
        (False, "signbert"),
        (False, "other"),
    )
    lr = {
        "clip": float(controlled_training["lr_clip"]),
        "signbert": float(controlled_training["lr_signbert"]),
        "other": float(controlled_training["lr_other"]),
    }
    weight_decay = float(controlled_training["weight_decay"])
    grouped = [
        {
            "params": [parameter for _, parameter in buckets[key]],
            "weight_decay": weight_decay if key[0] else 0.0,
            "lr": lr[key[1]],
        }
        for key in order
    ]
    owned = [id(parameter) for group in grouped for parameter in group["params"]]
    if len(owned) != len(named) or len(owned) != len(set(owned)):
        raise BaselineTrainingError("SEDS optimizer parameter ownership is incomplete/duplicated")
    optimizer_class = _load_pinned_optimizer(root)
    optimizer = optimizer_class(
        grouped,
        lr=lr["other"],
        warmup=float(controlled_training["warmup_fraction"]),
        schedule=str(controlled_training["schedule"]),
        b1=float(controlled_training["beta1"]),
        b2=float(controlled_training["beta2"]),
        e=float(controlled_training["epsilon"]),
        t_total=total_steps,
        weight_decay=weight_decay,
        max_grad_norm=float(controlled_training["optimizer_per_parameter_clip_norm"]),
    )
    manifest = {
        "schema_version": "seds_optimizer_manifest.v1",
        "optimizer": "pinned_bertadam",
        "source_path": str((root / "modules" / "optimization.py").resolve()),
        "source_sha256": hashlib.sha256(
            (root / "modules" / "optimization.py").read_bytes()
        ).hexdigest(),
        "total_steps": total_steps,
        "groups": [
            {
                "index": index,
                "decay": decay,
                "branch": family,
                "lr": lr[family],
                "weight_decay": weight_decay if decay else 0.0,
                "parameter_names": [name for name, _ in buckets[(decay, family)]],
            }
            for index, (decay, family) in enumerate(order)
        ],
    }
    return optimizer, manifest


def run_seds_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: SedsTrainingBatch,
    *,
    device: str | torch.device,
    global_grad_clip_norm: float,
) -> BaselineStepMetrics:
    if global_grad_clip_norm <= 0:
        raise BaselineTrainingError("global gradient clip norm must be positive")
    target = torch.device(device)
    video = _video_to(batch.video, target)
    text = _text_to(batch.text, target)
    augmented = _text_to(batch.augmented_text, target)
    right = {"pose": video.right_pose}
    left = {"pose": video.left_pose}
    body = {
        "pose": video.body_pose,
        "clips_start": video.clip_starts,
        "mask": video.legacy_video_mask,
        "rgb": video.rgb_features,
    }
    outputs = model(
        text.input_ids,
        text.token_type_ids,
        text.attention_mask,
        right,
        left,
        body,
        augmented.input_ids,
        augmented.attention_mask,
    )
    if not isinstance(outputs, tuple) or len(outputs) != 7:
        raise BaselineTrainingError("native SEDS objective must return exactly seven losses")
    loss = outputs[0]
    if not isinstance(loss, Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
        raise BaselineTrainingError("native SEDS total loss is not one finite tensor")
    values = []
    for value in outputs:
        scalar = float(value.detach()) if isinstance(value, Tensor) else float(value)
        if not math.isfinite(scalar):
            raise BaselineTrainingError("native SEDS objective returned a non-finite component")
        values.append(scalar)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), global_grad_clip_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad()
        raise BaselineTrainingError("native SEDS gradients contain NaN/Inf")
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        model.clip.logit_scale.clamp_(max=math.log(100.0))
    return BaselineStepMetrics(*values, global_grad_norm=float(grad_norm))


def _validate_batch_lineage(
    batch: SedsTrainingBatch | Any,
    records: Sequence[SampleRecord],
    native_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    video = batch.video if isinstance(batch, SedsTrainingBatch) else batch
    if video.pose_raw_frame_indices is None:
        raise BaselineTrainingError("native SEDS batch omitted pose-to-raw frame lineage")
    valid_counts = (video.legacy_video_mask == 0).sum(dim=1) - 1
    for index, (record, selected) in enumerate(
        zip(records, video.pose_raw_frame_indices, strict=True)
    ):
        row = native_rows[record.sample_id]
        valid_count = int(valid_counts[index])
        if list(selected) != row.get("selected_pose_raw_frame_indices") or video.clip_starts[
            index, :valid_count
        ].tolist() != row.get("clip_starts_in_selected_pose_steps"):
            raise BaselineTrainingError(
                f"native SEDS lineage differs from the registered audit: {record.sample_id}"
            )


def _verify_input_hashes(
    records: Sequence[SampleRecord],
    rows: Mapping[str, Mapping[str, Any]],
    *,
    pose_root: Path,
    rgb_root: Path,
) -> None:
    for record in records:
        row = rows[record.sample_id]
        if hash_seds_input(pose_root / str(record.pose_path)) != row.get(
            "pose_sha256"
        ) or hash_seds_input(rgb_root / str(record.rgb_feature_key)) != row.get("rgb_sha256"):
            raise BaselineTrainingError(
                f"native SEDS input changed after validation: {record.sample_id}"
            )


def _unique_text_records(records: Sequence[SampleRecord]) -> tuple[SampleRecord, ...]:
    ordered: dict[str, SampleRecord] = {}
    for record in records:
        previous = ordered.setdefault(record.text_id, record)
        if previous.text_model != record.text_model:
            raise BaselineTrainingError(f"inconsistent text for text ID: {record.text_id}")
    return tuple(ordered.values())


def _evaluate_dev(
    model: nn.Module,
    builder: SedsManifestInputBuilder,
    records: Sequence[SampleRecord],
    frame_maps: Mapping[str, CompactFrameMap],
    native_rows: Mapping[str, Mapping[str, Any]],
    relevance: Mapping[str, set[str]],
    *,
    upstream_root: Path,
    dual_mix: float,
    device: torch.device,
    batch_size: int,
    video_chunk: int,
    text_chunk: int,
    topk: Sequence[int],
) -> tuple[RetrievalMetrics, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    adapter = SedsAdapter(model, upstream_root=upstream_root, dual_mix=dual_mix)
    text_records = _unique_text_records(records)
    video_parts: list[NativeVideoFeatures] = []
    text_parts: list[NativeTextFeatures] = []
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                part = records[start : start + batch_size]
                native = builder.build_video_batch(
                    part,
                    raw_frame_counts=[
                        frame_maps[item.sample_id].video_frame_count for item in part
                    ],
                    frames_per_second=[frame_maps[item.sample_id].fps for item in part],
                )
                _validate_batch_lineage(native, part, native_rows)
                video_parts.append(
                    _cpu_video(adapter.encode_video_native(_video_to(native, device)))
                )
            for start in range(0, len(text_records), batch_size):
                part = text_records[start : start + batch_size]
                native = builder.build_text_batch(part)
                text_parts.append(_cpu_text(adapter.encode_text_native(_text_to(native, device))))
            video = _join_video(video_parts)
            text = _join_text(text_parts)
            scores = score_feature_gallery(
                adapter,
                video,
                text,
                device=device,
                video_chunk=video_chunk,
                text_chunk=text_chunk,
            )
    finally:
        model.train()
    video_ids = tuple(item.video_id for item in records)
    metrics = evaluate_retrieval(scores, video_ids, text.text_ids, relevance, topk=topk)
    return metrics, scores, video_ids, text.text_ids


def _repository_revision(root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise BaselineTrainingError("cannot resolve implementation Git revision") from exc
    if not revision or dirty:
        raise BaselineTrainingError("controlled baseline training requires a clean Git revision")
    return revision


def _emit_progress(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True), file=sys.stderr, flush=True)


def _atomic_model_state(path: Path, model: nn.Module) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, temporary)
    return hash_artifact(temporary)[0]


def _repair_best_output(path: Path, expected_sha256: str) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    if path.is_file() and hash_artifact(path)[0] == expected_sha256:
        if pending.exists():
            pending.unlink()
        return
    if pending.is_file() and hash_artifact(pending)[0] == expected_sha256:
        os.replace(pending, path)
        return
    raise BaselineTrainingError(f"best-output transaction cannot be recovered: {path}")


def _resolve_parent_inputs(
    config: Mapping[str, Any], resolver: ArtifactResolver
) -> tuple[ResolvedArtifact, ResolvedArtifact]:
    try:
        return (
            resolver.resolve("validate_data", "audit", scope="shared"),
            resolver.resolve("validate_data", "native_frame_maps_dir", scope="shared"),
        )
    except ArtifactError as exc:
        raise BaselineTrainingError(
            "MISSING_PARENT_ARTIFACT: validate_data.audit/native_frame_maps_dir must pass "
            "before baseline training"
        ) from exc


def train_seds_baseline(
    config: Mapping[str, Any],
    *,
    batch_size: int = 128,
    eval_batch_size: int = 64,
    device: str = "cuda:0",
    resume: bool = False,
) -> dict[str, Any]:
    """Train and dev-select the controlled native SEDS B0; never read the test split."""
    data = _mapping(config, "data")
    baseline = _mapping(config, "baseline")
    evaluation = _mapping(config, "evaluation")
    if baseline.get("family") != "seds":
        raise BaselineTrainingError("baseline training currently requires family=seds")
    if baseline.get("locked_checkpoint") is not None:
        raise BaselineTrainingError(
            "baseline.locked_checkpoint must be null until controlled dev selection completes"
        )
    if batch_size <= 0 or eval_batch_size <= 0:
        raise BaselineTrainingError("training/evaluation batch sizes must be positive")
    resolver = ArtifactResolver(config)
    audit_parent, native_parent = _resolve_parent_inputs(config, resolver)
    output_dir = resolver.output_path("shared", "baseline", "train")
    latest = output_dir / "latest.pt"
    latest_checksum = output_dir / "latest.pt.sha256"
    best_model = output_dir / "best_model.bin"
    best_scores = output_dir / "best_dev_scores.npz"
    progress_path = output_dir / "progress.json"
    report_path = output_dir / "training.json"
    outputs = (latest, latest_checksum, best_model, best_scores, progress_path, report_path)
    pending_outputs = (
        best_model.with_suffix(best_model.suffix + ".pending"),
        best_scores.with_suffix(best_scores.suffix + ".pending"),
    )
    if not resume and any(path.exists() for path in (*outputs, *pending_outputs)):
        raise BaselineTrainingError(
            "baseline training outputs already exist; use --resume after verifying provenance"
        )
    upstream_root = _required_path(data.get("upstream_root"), "data.upstream_root")
    reproduction_path = _required_path(
        baseline.get("reproduction_config"), "baseline.reproduction_config"
    )
    reproduction = load_seds_reproduction(reproduction_path, upstream_root=upstream_root)
    training = reproduction.controlled_training
    if batch_size != int(training["effective_batch_size"]):
        raise BaselineTrainingError(
            f"controlled SEDS batch size must be {training['effective_batch_size']}"
        )
    if int(training["world_size"]) != 1 or int(training["gradient_accumulation_steps"]) != 1:
        raise BaselineTrainingError("controlled SEDS runner requires world_size=accumulation=1")
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise BaselineTrainingError("controlled native SEDS training requires CUDA")
    repository_root = Path(__file__).resolve().parents[4]
    revision = _repository_revision(repository_root)
    _required_path(baseline.get("initial_weights"), "baseline.initial_weights")
    _required_path(
        str(upstream_root / reproduction.external_assets["clip_initialization"]),
        "baseline.clip_initialization",
    )

    train_manifest = _configured_or_prepared(data, resolver, "train_manifest", "train_manifest")
    dev_manifest = _configured_or_prepared(data, resolver, "dev_manifest", "dev_manifest")
    frame_maps_dir = _configured_or_prepared(data, resolver, "frame_maps_dir", "frame_maps_dir")
    relevance_dir = _configured_or_prepared(data, resolver, "relevance_dir", "relevance_dir")
    pose_root = _required_path(data.get("pose_root"), "data.pose_root")
    rgb_root = _required_path(data.get("rgb_cache_root"), "data.rgb_cache_root")
    train_records = load_manifest(train_manifest, expected_split="train")
    dev_records = load_manifest(dev_manifest, expected_split="dev")
    train_maps_seq = load_compact_frame_maps(
        frame_maps_dir / "train.jsonl",
        expected_sample_ids=[item.sample_id for item in train_records],
    )
    dev_maps_seq = load_compact_frame_maps(
        frame_maps_dir / "dev.jsonl",
        expected_sample_ids=[item.sample_id for item in dev_records],
    )
    train_maps = {item.sample_id: item for item in train_maps_seq}
    dev_maps = {item.sample_id: item for item in dev_maps_seq}
    train_rows = _load_native_frame_rows(native_parent.path / "train.jsonl", train_records)
    dev_rows = _load_native_frame_rows(native_parent.path / "dev.jsonl", dev_records)
    _verify_input_hashes(train_records, train_rows, pose_root=pose_root, rgb_root=rgb_root)
    _verify_input_hashes(dev_records, dev_rows, pose_root=pose_root, rgb_root=rgb_root)
    dev_text_ids = tuple(item.text_id for item in _unique_text_records(dev_records))
    dev_relevance_path = relevance_dir / "dev.jsonl"
    dev_relevance = load_relevance(
        dev_relevance_path,
        video_ids=[item.video_id for item in dev_records],
        text_ids=dev_text_ids,
    )

    seed = int(config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model, initialization = SedsAdapter.build_official_training_model(
        config, upstream_root=upstream_root, device=target_device
    )
    text_count = len({item.text_id for item in train_records})
    steps_per_epoch = math.ceil(text_count / batch_size)
    epochs = int(training["epochs"])
    total_steps = steps_per_epoch * epochs
    optimizer, optimizer_manifest = build_seds_optimizer(
        model,
        upstream_root=upstream_root,
        controlled_training=training,
        total_steps=total_steps,
    )
    optimizer.zero_grad()
    train_builder = SedsManifestInputBuilder(
        upstream_root=upstream_root,
        reproduction_config=reproduction_path,
        pose_root=pose_root,
        rgb_root=rgb_root,
        cache_video_samples=True,
    )
    dev_builder = SedsManifestInputBuilder(
        upstream_root=upstream_root,
        reproduction_config=reproduction_path,
        pose_root=pose_root,
        rgb_root=rgb_root,
        cache_video_samples=True,
    )

    fingerprints = {
        "implementation_git_revision": revision,
        "data_audit": audit_parent.record.sha256,
        "native_frame_maps": native_parent.record.sha256,
        "train_manifest": manifest_hash(train_manifest),
        "dev_manifest": manifest_hash(dev_manifest),
        "dev_relevance": relevance_hash(dev_relevance_path),
        "reproduction_config": hashlib.sha256(reproduction_path.read_bytes()).hexdigest(),
        "clip_initialization": str(initialization["clip_initialization_sha256"]),
        "signbert_initialization": str(initialization["signbert_initialization_sha256"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_sha = config_hash(config)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    start_epoch = 1
    global_step = 0
    if resume:
        try:
            state = load_training_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                expected_config_hash=configuration_sha,
                expected_fingerprints=fingerprints,
            )
        except CheckpointError as exc:
            raise BaselineTrainingError(f"controlled SEDS resume failed: {exc}") from exc
        start_epoch = state.epoch + 1
        global_step = state.global_step
        best = state.best_dev
        raw_history = state.sampler_state.get("history")
        if not isinstance(raw_history, list) or len(raw_history) != state.epoch:
            raise BaselineTrainingError("resume checkpoint history is inconsistent")
        history = list(raw_history)
        if best:
            _repair_best_output(best_model, str(best["checkpoint_sha256"]))
            _repair_best_output(best_scores, str(best["scores_sha256"]))
    if start_epoch > epochs:
        raise BaselineTrainingError("resume checkpoint already completed controlled training")

    for epoch in range(start_epoch, epochs + 1):
        plan = build_baseline_epoch_plan(train_records, seed=seed, epoch=epoch)
        augmentation = random.Random(plan.augmentation_seed)
        totals = np.zeros(8, dtype=np.float64)
        observed_steps = 0
        for start in range(0, len(plan.records), batch_size):
            part = plan.records[start : start + batch_size]
            batch = train_builder.build_training_batch(
                part,
                raw_frame_counts=[train_maps[item.sample_id].video_frame_count for item in part],
                frames_per_second=[train_maps[item.sample_id].fps for item in part],
                generator=augmentation,
            )
            _validate_batch_lineage(batch, part, train_rows)
            step_metrics = run_seds_training_step(
                model,
                optimizer,
                batch,
                device=target_device,
                global_grad_clip_norm=float(training["global_grad_clip_norm"]),
            )
            totals += np.asarray(list(asdict(step_metrics).values()), dtype=np.float64)
            observed_steps += 1
            global_step += 1
            if observed_steps % 10 == 0 or observed_steps == steps_per_epoch:
                _emit_progress(
                    {
                        "stage": "baseline_train",
                        "epoch": epoch,
                        "epoch_step": observed_steps,
                        "steps_per_epoch": steps_per_epoch,
                        "global_step": global_step,
                        "loss": step_metrics.loss,
                    }
                )
        if observed_steps != steps_per_epoch:
            raise BaselineTrainingError("observed SEDS optimizer steps differ from contract")
        metrics, scores, video_ids, text_ids = _evaluate_dev(
            model,
            dev_builder,
            dev_records,
            dev_maps,
            dev_rows,
            dev_relevance,
            upstream_root=upstream_root,
            dual_mix=float(baseline["dual_mix"]),
            device=target_device,
            batch_size=eval_batch_size,
            video_chunk=int(evaluation["query_chunk"]),
            text_chunk=int(evaluation["candidate_chunk"]),
            topk=tuple(int(item) for item in evaluation["topk"]),
        )
        candidate = DevCandidate(
            epoch=epoch,
            optimizer_step=global_step,
            t2v_r1=metrics.t2v.recall[1],
            v2t_r1=metrics.v2t.recall[1],
            checkpoint_path=str(best_model),
        )
        existing = None
        if best:
            existing = DevCandidate(
                epoch=int(best["epoch"]),
                optimizer_step=int(best["optimizer_step"]),
                t2v_r1=float(best["t2v_r1"]),
                v2t_r1=float(best["v2t_r1"]),
                checkpoint_path=str(best_model),
            )
        improved = (
            existing is None
            or select_earliest_best([existing, candidate], split="dev") == candidate
        )
        model_sha = scores_sha = None
        if improved:
            model_sha = _atomic_model_state(best_model, model)
            pending_scores = best_scores.with_suffix(best_scores.suffix + ".pending")
            _atomic_scores(pending_scores, scores, video_ids, text_ids)
            scores_sha = hash_artifact(pending_scores)[0]
            best = {
                **asdict(candidate),
                "endpoint": candidate.endpoint,
                "checkpoint_sha256": model_sha,
                "scores_sha256": scores_sha,
            }
        epoch_record = {
            "epoch": epoch,
            "optimizer_step": global_step,
            "plan_sha256": plan.sha256,
            "sample_count": len(plan.records),
            "optimizer_steps": observed_steps,
            "mean_step_metrics": dict(
                zip(
                    BaselineStepMetrics.__dataclass_fields__,
                    (float(value) for value in totals / observed_steps),
                    strict=True,
                )
            ),
            "dev": metrics.to_dict(),
            "selection_endpoint": candidate.endpoint,
            "selected": improved,
        }
        history.append(epoch_record)
        save_training_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            epoch=epoch,
            global_step=global_step,
            best_dev=best,
            sampler_state={"last_plan_sha256": plan.sha256, "history": history},
            fingerprints=fingerprints,
            config_hash=configuration_sha,
            git_revision=revision,
            optimizer_manifest=optimizer_manifest,
        )
        if improved:
            assert model_sha is not None and scores_sha is not None
            _repair_best_output(best_model, model_sha)
            _repair_best_output(best_scores, scores_sha)
        _atomic_json(
            progress_path,
            {
                "schema_version": "seds_baseline_progress.v1",
                "completed_epoch": epoch,
                "global_step": global_step,
                "best_dev": best,
                "history": history,
            },
        )
        _emit_progress(
            {
                "stage": "baseline_train_dev",
                "epoch": epoch,
                "global_step": global_step,
                "t2v_r1": candidate.t2v_r1,
                "v2t_r1": candidate.v2t_r1,
                "selection_endpoint": candidate.endpoint,
                "selected": improved,
            }
        )

    if len(history) != epochs or global_step != total_steps or not best:
        raise BaselineTrainingError("controlled SEDS training did not complete its fixed budget")
    report: dict[str, Any] = {
        "schema_version": "seds_baseline_training.v1",
        "ready": True,
        "config_sha256": configuration_sha,
        "implementation_git_revision": revision,
        "test_content_used_for_tuning": False,
        "controlled_training": dict(training),
        "initialization": dict(initialization),
        "fingerprints": fingerprints,
        "train": {
            "manifest": str(train_manifest),
            "manifest_sha256": manifest_hash(train_manifest),
            "record_count": len(train_records),
            "unique_text_count": text_count,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "total_optimizer_steps": total_steps,
        },
        "dev": {
            "manifest": str(dev_manifest),
            "manifest_sha256": manifest_hash(dev_manifest),
            "relevance": str(dev_relevance_path),
            "relevance_sha256": relevance_hash(dev_relevance_path),
            "record_count": len(dev_records),
            "text_count": len(dev_text_ids),
        },
        "optimizer": optimizer_manifest,
        "selection": best,
        "history": history,
        "retained_checkpoints": [str(best_model), str(latest)],
    }
    _atomic_json(report_path, report)
    registered = resolver.record_stage(
        "baseline_train",
        {
            "best_dev_scores": best_scores,
            "locked_checkpoint": best_model,
            "report": report_path,
            "progress": progress_path,
            "resume_checkpoint": latest,
            "resume_checksum": latest_checksum,
        },
        scope="shared",
        parents=(audit_parent, native_parent),
        metadata={
            "selection_split": "dev",
            "selection_metric": "mean_t2v_v2t_r1",
            "optimizer_steps": total_steps,
        },
    )
    report["artifacts"] = {
        name: {"artifact_id": item.artifact_id, "sha256": item.sha256}
        for name, item in registered.items()
    }
    return report
