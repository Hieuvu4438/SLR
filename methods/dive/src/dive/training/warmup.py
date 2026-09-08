from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dive.eval.metrics import RetrievalMetrics, evaluate_retrieval
from dive.losses import retrieval_loss
from dive.models.evidence import EvidenceEncoder, detach_frozen_input, state_hash
from dive.models.scoring import evidence_score_block
from dive.training.optimizer import (
    build_evidence_optimizer,
    build_warmup_cosine_scheduler,
    optimizer_settings,
)
from dive.training.selection import DevCandidate, select_earliest_best
from dive.training.state import load_training_checkpoint, save_training_checkpoint


class WarmupContractError(ValueError):
    """Warm-up data, metrics, or artifact contracts are invalid."""


@dataclass(frozen=True)
class WarmupBatch:
    pose: Any
    rgb_local: Tensor
    grid: Tensor
    video_mask: Tensor
    text_units: Tensor
    text_mask: Tensor
    positives: Tensor
    candidates: Tensor


@dataclass(frozen=True)
class WarmupGallery:
    batch: WarmupBatch
    video_ids: tuple[str, ...]
    text_ids: tuple[str, ...]
    video_to_text_positives: Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class WarmupVideoGalleryBatch:
    pose: Any
    rgb_local: Tensor
    grid: Tensor
    video_mask: Tensor
    video_ids: tuple[str, ...]


@dataclass(frozen=True)
class WarmupTextGalleryBatch:
    text_units: Tensor
    text_mask: Tensor
    text_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChunkedWarmupGallery:
    """Restartable dev streams whose cross product is scored one bounded block at a time."""

    video_batches: Callable[[], Iterable[WarmupVideoGalleryBatch]]
    text_batches: Callable[[], Iterable[WarmupTextGalleryBatch]]
    video_ids: tuple[str, ...]
    text_ids: tuple[str, ...]
    video_to_text_positives: Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class WarmupResult:
    winner: DevCandidate
    candidates: tuple[DevCandidate, ...]
    reference_path: str
    reference_checksum: str
    reference_state_hash: str
    selection_path: str
    optimizer_manifest: Mapping[str, Any]


def _validate_batch(batch: WarmupBatch) -> None:
    num_videos = batch.rgb_local.shape[0]
    num_texts = batch.text_units.shape[0]
    if batch.rgb_local.shape[:2] != batch.video_mask.shape:
        raise WarmupContractError("RGB features and video mask disagree")
    if batch.grid.shape[:2] != batch.video_mask.shape:
        raise WarmupContractError("canonical grid and video mask disagree")
    if batch.text_units.shape[:2] != batch.text_mask.shape:
        raise WarmupContractError("text units and text mask disagree")
    if batch.video_mask.dtype != torch.bool or batch.text_mask.dtype != torch.bool:
        raise WarmupContractError("warm-up masks must be bool with True=valid")
    expected = (num_videos, num_texts)
    if batch.positives.shape != expected or batch.candidates.shape != expected:
        raise WarmupContractError("positive/candidate relations have the wrong score layout")
    if batch.positives.dtype != torch.bool or batch.candidates.dtype != torch.bool:
        raise WarmupContractError("positive/candidate relations must be bool")
    if bool((batch.positives & ~batch.candidates).any()):
        raise WarmupContractError("known positives must remain retrieval candidates")
    if num_videos and not bool(batch.positives.any(dim=1).all()):
        raise WarmupContractError("every warm-up video query requires a known positive")
    if num_texts and not bool(batch.positives.any(dim=0).all()):
        raise WarmupContractError("every warm-up text query requires a known positive")


def run_warmup_step(
    model: EvidenceEncoder,
    optimizer: torch.optim.Optimizer,
    batch: WarmupBatch,
    *,
    tau_alignment: float,
    tau_retrieval: float,
    grad_clip_norm: float,
) -> tuple[float, float]:
    """Optimize only symmetric E_local retrieval on an ordinary training batch."""
    _validate_batch(batch)
    if grad_clip_norm <= 0:
        raise WarmupContractError("gradient clip norm must be positive")
    model.train()
    model.enforce_frozen_batch_norm()
    optimizer.zero_grad(set_to_none=True)
    local = model(
        detach_frozen_input(batch.pose),
        batch.rgb_local.detach(),
        batch.grid,
        batch.video_mask,
    )
    scores, _ = evidence_score_block(
        local,
        batch.text_units.detach(),
        batch.video_mask,
        batch.text_mask,
        tau_alignment,
    )
    loss = retrieval_loss(scores, batch.positives, batch.candidates, tau_retrieval)
    if not torch.isfinite(loss):
        raise FloatingPointError("warm-up retrieval loss is nonfinite")
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or not any(parameter.grad is not None for parameter in trainable):
        raise WarmupContractError("warm-up evidence model received no gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    if not torch.isfinite(gradient_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("warm-up gradients contain NaN/Inf")
    optimizer.step()
    return float(loss.detach()), float(gradient_norm)


def evaluate_warmup_gallery(
    model: EvidenceEncoder,
    gallery: WarmupGallery,
    *,
    tau_alignment: float,
    topk: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    _validate_batch(gallery.batch)
    if len(gallery.video_ids) != gallery.batch.rgb_local.shape[0]:
        raise WarmupContractError("dev video IDs do not cover the gallery")
    if len(gallery.text_ids) != gallery.batch.text_units.shape[0]:
        raise WarmupContractError("dev text IDs do not cover the gallery")
    model.eval()
    with torch.no_grad():
        local = model(
            detach_frozen_input(gallery.batch.pose),
            gallery.batch.rgb_local.detach(),
            gallery.batch.grid,
            gallery.batch.video_mask,
        )
        scores, _ = evidence_score_block(
            local,
            gallery.batch.text_units.detach(),
            gallery.batch.video_mask,
            gallery.batch.text_mask,
            tau_alignment,
        )
    return evaluate_retrieval(
        scores.float().cpu().numpy(),
        gallery.video_ids,
        gallery.text_ids,
        gallery.video_to_text_positives,
        topk=topk,
    )


def _validate_gallery_ids(ids: Sequence[str], expected_count: int, name: str) -> None:
    if len(ids) != expected_count:
        raise WarmupContractError(f"{name} IDs do not match their batch size")
    if any(not isinstance(item, str) or not item for item in ids):
        raise WarmupContractError(f"{name} IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise WarmupContractError(f"{name} IDs contain duplicates within a batch")


def _validate_video_gallery_batch(batch: WarmupVideoGalleryBatch) -> None:
    if batch.rgb_local.ndim != 3:
        raise WarmupContractError("gallery RGB features must have shape [B,N,D]")
    if batch.video_mask.dtype != torch.bool or batch.video_mask.shape != batch.rgb_local.shape[:2]:
        raise WarmupContractError("gallery video mask must be bool [B,N]")
    if batch.grid.shape[:2] != batch.video_mask.shape:
        raise WarmupContractError("gallery grid and video mask disagree")
    _validate_gallery_ids(batch.video_ids, len(batch.rgb_local), "gallery video")


def _validate_text_gallery_batch(batch: WarmupTextGalleryBatch) -> None:
    if batch.text_units.ndim != 3:
        raise WarmupContractError("gallery text units must have shape [B,M,D]")
    if batch.text_mask.dtype != torch.bool or batch.text_mask.shape != batch.text_units.shape[:2]:
        raise WarmupContractError("gallery text mask must be bool [B,M]")
    _validate_gallery_ids(batch.text_ids, len(batch.text_units), "gallery text")


def evaluate_chunked_warmup_gallery(
    model: EvidenceEncoder,
    gallery: ChunkedWarmupGallery,
    *,
    tau_alignment: float,
    topk: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Evaluate the exact full gallery without constructing its 4-D interaction tensor."""
    if not gallery.video_ids or not gallery.text_ids:
        raise WarmupContractError("chunked dev gallery cannot be empty")
    if len(gallery.video_ids) != len(set(gallery.video_ids)):
        raise WarmupContractError("chunked gallery video IDs contain duplicates")
    if len(gallery.text_ids) != len(set(gallery.text_ids)):
        raise WarmupContractError("chunked gallery text IDs contain duplicates")
    model.eval()
    score_rows: list[Tensor] = []
    observed_video_ids: list[str] = []
    canonical_text_ids: tuple[str, ...] | None = None
    with torch.no_grad():
        for video_batch in gallery.video_batches():
            _validate_video_gallery_batch(video_batch)
            local = model(
                detach_frozen_input(video_batch.pose),
                video_batch.rgb_local.detach(),
                video_batch.grid,
                video_batch.video_mask,
            )
            row_blocks: list[Tensor] = []
            observed_text_ids: list[str] = []
            for text_batch in gallery.text_batches():
                _validate_text_gallery_batch(text_batch)
                scores, valid = evidence_score_block(
                    local,
                    text_batch.text_units.detach(),
                    video_batch.video_mask,
                    text_batch.text_mask,
                    tau_alignment,
                )
                if not bool(valid.all()):
                    raise WarmupContractError("dev gallery contains an all-invalid video or text")
                row_blocks.append(scores.float().cpu())
                observed_text_ids.extend(text_batch.text_ids)
            if not row_blocks:
                raise WarmupContractError("chunked dev gallery yielded no text batches")
            current_text_ids = tuple(observed_text_ids)
            if canonical_text_ids is None:
                canonical_text_ids = current_text_ids
            elif current_text_ids != canonical_text_ids:
                raise WarmupContractError("text batch stream is not restartable and deterministic")
            score_rows.append(torch.cat(row_blocks, dim=1))
            observed_video_ids.extend(video_batch.video_ids)
    if not score_rows:
        raise WarmupContractError("chunked dev gallery yielded no video batches")
    if tuple(observed_video_ids) != gallery.video_ids:
        raise WarmupContractError("video batch stream does not exactly cover declared gallery IDs")
    if canonical_text_ids != gallery.text_ids:
        raise WarmupContractError("text batch stream does not exactly cover declared gallery IDs")
    return evaluate_retrieval(
        torch.cat(score_rows, dim=0).numpy(),
        gallery.video_ids,
        gallery.text_ids,
        gallery.video_to_text_positives,
        topk=topk,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dev_gallery_fingerprint(gallery: WarmupGallery | ChunkedWarmupGallery) -> str:
    relevance = {
        str(video_id): sorted(str(text_id) for text_id in text_ids)
        for video_id, text_ids in sorted(gallery.video_to_text_positives.items())
    }
    payload = {
        "video_ids": list(gallery.video_ids),
        "text_ids": list(gallery.text_ids),
        "video_to_text_positives": relevance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _export_reference(
    path: Path,
    model: EvidenceEncoder,
    *,
    winner: DevCandidate,
    checkpoint_checksum: str,
    config_hash: str,
    fingerprints: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    git_revision: str,
) -> tuple[str, str]:
    model.eval()
    model.requires_grad_(False)
    for name, value in model.state_dict().items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise WarmupContractError(
                f"correctness reference state must be float32, found {value.dtype} at {name}"
            )
    reference_hash = state_hash(model)
    payload = {
        "schema_version": "dive_reference.v1",
        "model": model.state_dict(),
        "state_hash": reference_hash,
        "selection": asdict(winner),
        "source_checkpoint_sha256": checkpoint_checksum,
        "config_hash": config_hash,
        "resolved_config": dict(resolved_config),
        "fingerprints": dict(fingerprints),
        "dtype": "float32",
        "git_revision": git_revision,
        "torch_version": torch.__version__,
        "selection_split": "dev",
        "selection_metric": "mean_t2v_v2t_r1",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    checksum = _sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.write_text(checksum + "\n", encoding="ascii")
    os.replace(sidecar_tmp, sidecar)
    return reference_hash, checksum


def run_evidence_warmup(
    model: EvidenceEncoder,
    train_batches: Callable[[int], Iterable[WarmupBatch]],
    dev_gallery: WarmupGallery | ChunkedWarmupGallery,
    *,
    config: Mapping[str, Any],
    steps_per_epoch: int,
    output_dir: str | Path,
    config_hash: str,
    fingerprints: Mapping[str, str],
    git_revision: str,
    forbidden_modules: Sequence[torch.nn.Module] = (),
    resume: bool = False,
) -> WarmupResult:
    """Run the fixed-budget evidence warm-up and export the dev-selected reference."""
    train = config["train"]
    evidence = config["evidence"]
    loss_config = config["loss"]
    epochs = int(train["warmup_epochs"])
    if epochs <= 0 or steps_per_epoch <= 0:
        raise WarmupContractError("warm-up epochs and steps_per_epoch must be positive")
    if bool(train["amp"]):
        raise WarmupContractError("correctness warm-up does not permit AMP")
    required_fingerprints = {"baseline", "data", "grid", "units"}
    missing_fingerprints = required_fingerprints - set(fingerprints)
    if missing_fingerprints or any(not str(value) for value in fingerprints.values()):
        raise WarmupContractError(
            f"warm-up fingerprints are missing/empty: {sorted(missing_fingerprints)}"
        )
    run_fingerprints = dict(fingerprints)
    run_fingerprints["dev_gallery"] = _dev_gallery_fingerprint(dev_gallery)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = build_evidence_optimizer(
        model,
        forbidden_modules=forbidden_modules,
        **optimizer_settings(config),
    )
    total_steps = epochs * steps_per_epoch
    scheduler = build_warmup_cosine_scheduler(
        bundle.optimizer,
        total_steps=total_steps,
        warmup_fraction=float(train["warmup_fraction"]),
        minimum_lr_fraction=float(train["minimum_lr_fraction"]),
    )
    candidates: list[DevCandidate] = []
    global_step = 0
    start_epoch = 1
    existing_checkpoints = sorted(destination.glob("warmup_epoch_*.pt"))
    if resume:
        if not existing_checkpoints:
            raise WarmupContractError("warm-up resume requested but no epoch checkpoint exists")
        latest = existing_checkpoints[-1]
        try:
            latest_epoch = int(latest.stem.removeprefix("warmup_epoch_"))
        except ValueError as exc:
            raise WarmupContractError("warm-up checkpoint name has an invalid epoch") from exc
        state = load_training_checkpoint(
            latest,
            model=model,
            optimizer=bundle.optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_config_hash=config_hash,
            expected_fingerprints=run_fingerprints,
        )
        if state.epoch != latest_epoch:
            raise WarmupContractError("warm-up checkpoint epoch differs from its file name")
        raw_candidates = state.sampler_state.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) != state.epoch:
            raise WarmupContractError("warm-up resume candidate history is incomplete")
        try:
            candidates = [DevCandidate(**dict(item)) for item in raw_candidates]
        except (TypeError, ValueError) as exc:
            raise WarmupContractError("warm-up resume candidate history is invalid") from exc
        global_step = state.global_step
        start_epoch = state.epoch + 1
    elif existing_checkpoints or (destination / "reference.pt").exists():
        raise WarmupContractError(
            "warm-up output already contains checkpoints; pass resume to continue"
        )
    for epoch in range(start_epoch, epochs + 1):
        observed_steps = 0
        total_loss = 0.0
        total_gradient_norm = 0.0
        for batch in train_batches(epoch):
            observed_steps += 1
            if observed_steps > steps_per_epoch:
                raise WarmupContractError(f"epoch {epoch} yielded too many optimizer steps")
            loss, gradient_norm = run_warmup_step(
                model,
                bundle.optimizer,
                batch,
                tau_alignment=float(evidence["tau_alignment"]),
                tau_retrieval=float(loss_config["tau_retrieval"]),
                grad_clip_norm=float(train["grad_clip_norm"]),
            )
            scheduler.step()
            global_step += 1
            total_loss += loss
            total_gradient_norm += gradient_norm
        if observed_steps != steps_per_epoch:
            raise WarmupContractError(
                f"epoch {epoch} yielded {observed_steps}, expected {steps_per_epoch} optimizer steps"
            )
        evaluate = (
            evaluate_chunked_warmup_gallery
            if isinstance(dev_gallery, ChunkedWarmupGallery)
            else evaluate_warmup_gallery
        )
        metrics = evaluate(
            model,
            dev_gallery,
            tau_alignment=float(evidence["tau_alignment"]),
            topk=tuple(int(item) for item in config["evaluation"]["topk"]),
        )
        checkpoint = destination / f"warmup_epoch_{epoch:03d}.pt"
        candidate = DevCandidate(
            epoch=epoch,
            optimizer_step=global_step,
            t2v_r1=metrics.t2v.recall[1],
            v2t_r1=metrics.v2t.recall[1],
            checkpoint_path=str(checkpoint),
        )
        candidates.append(candidate)
        best_so_far = select_earliest_best(candidates, split="dev")
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=bundle.optimizer,
            scheduler=scheduler,
            scaler=None,
            epoch=epoch,
            global_step=global_step,
            best_dev=asdict(best_so_far),
            sampler_state={
                "stage": "warmup",
                "epoch_step": observed_steps,
                "mean_train_loss": total_loss / observed_steps,
                "mean_gradient_norm": total_gradient_norm / observed_steps,
                "candidates": [asdict(item) for item in candidates],
            },
            fingerprints=run_fingerprints,
            config_hash=config_hash,
            git_revision=git_revision,
            optimizer_manifest=bundle.manifest,
        )

    winner = select_earliest_best(candidates, split="dev")
    load_training_checkpoint(
        winner.checkpoint_path,
        model=model,
        optimizer=bundle.optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_config_hash=config_hash,
        expected_fingerprints=run_fingerprints,
        restore_rng=False,
    )
    winner_checkpoint = Path(winner.checkpoint_path)
    checkpoint_checksum = (
        winner_checkpoint.with_suffix(winner_checkpoint.suffix + ".sha256")
        .read_text(encoding="ascii")
        .strip()
    )
    reference_path = destination / "reference.pt"
    reference_hash, reference_checksum = _export_reference(
        reference_path,
        model,
        winner=winner,
        checkpoint_checksum=checkpoint_checksum,
        config_hash=config_hash,
        fingerprints=run_fingerprints,
        resolved_config=config,
        git_revision=git_revision,
    )
    selection_path = destination / "selection.json"
    _atomic_json(
        selection_path,
        {
            "schema_version": "dive_warmup_selection.v1",
            "selection_split": "dev",
            "selection_metric": "mean_t2v_v2t_r1",
            "tie_break": "earliest_optimizer_step",
            "winner": asdict(winner),
            "candidates": [asdict(item) | {"endpoint": item.endpoint} for item in candidates],
            "reference_path": str(reference_path),
            "reference_sha256": reference_checksum,
            "reference_state_hash": reference_hash,
            "config_hash": config_hash,
            "fingerprints": run_fingerprints,
            "teacher_only": True,
        },
    )
    return WarmupResult(
        winner=winner,
        candidates=tuple(candidates),
        reference_path=str(reference_path),
        reference_checksum=reference_checksum,
        reference_state_hash=reference_hash,
        selection_path=str(selection_path),
        optimizer_manifest=bundle.manifest,
    )
