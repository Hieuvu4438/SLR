"""Leakage-safe planning for P14T train-only I3D adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class AdaptationPlanError(ValueError):
    """Raised when adaptation provenance is incomplete or leaks forbidden IDs."""


def _manifest_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AdaptationPlanError(f"cannot read manifest {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdaptationPlanError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(record, Mapping) or not record.get("sample_id"):
            raise AdaptationPlanError(f"missing sample_id at {path}:{line_number}")
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise AdaptationPlanError(f"duplicate sample_id {sample_id!r} in {path}")
        seen.add(sample_id)
        ids.append(sample_id)
    if not ids:
        raise AdaptationPlanError(f"manifest is empty: {path}")
    return ids


def _class_vocabulary(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AdaptationPlanError(f"cannot read class vocabulary {path}: {error}") from error
    names: list[str] = []
    for index, line in enumerate(lines):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) != index:
            raise AdaptationPlanError(f"class vocabulary must have contiguous labels at line {index + 1}")
        names.append(fields[1])
    if not names or len(set(names)) != len(names):
        raise AdaptationPlanError("class vocabulary must be non-empty and unique")
    return names


def _fold(sample_id: str, seed: int, holdout_modulus: int) -> str:
    digest = hashlib.sha256(f"ocem-p14t-adaptation-v1\0{seed}\0{sample_id}".encode()).digest()
    return "holdout" if int.from_bytes(digest[:8], "big") % holdout_modulus == 0 else "train"


def build_p14t_adaptation_plan(
    *,
    train_manifest: str | Path,
    forbidden_manifests: Sequence[str | Path],
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    class_vocabulary: str | Path,
    seed: int = 0,
    holdout_modulus: int = 10,
) -> dict[str, Any]:
    """Resolve exact train/holdout IDs and pin the observable CiCo recipe."""

    if seed < 0 or holdout_modulus < 2:
        raise AdaptationPlanError("seed must be nonnegative and holdout_modulus at least 2")
    train_manifest = Path(train_manifest)
    forbidden_paths = [Path(path) for path in forbidden_manifests]
    checkpoint = Path(checkpoint)
    class_vocabulary = Path(class_vocabulary)
    if not checkpoint.is_file():
        raise AdaptationPlanError(f"checkpoint is missing: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise AdaptationPlanError(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha256} != {expected_checkpoint_sha256}"
        )

    source_ids = _manifest_ids(train_manifest)
    forbidden_by_manifest = {str(path.resolve()): _manifest_ids(path) for path in forbidden_paths}
    forbidden_ids = {sample_id for ids in forbidden_by_manifest.values() for sample_id in ids}
    leaked = sorted(set(source_ids) & forbidden_ids)
    if leaked:
        raise AdaptationPlanError(f"train manifest overlaps forbidden IDs: {leaked[:10]}")

    vocabulary = _class_vocabulary(class_vocabulary)
    assignments = [
        {"sample_id": sample_id, "adaptation_split": _fold(sample_id, seed, holdout_modulus)}
        for sample_id in sorted(source_ids)
    ]
    train_ids = [item["sample_id"] for item in assignments if item["adaptation_split"] == "train"]
    holdout_ids = [
        item["sample_id"] for item in assignments if item["adaptation_split"] == "holdout"
    ]
    if not train_ids or not holdout_ids:
        raise AdaptationPlanError("deterministic train/holdout split produced an empty partition")

    return {
        "schema_version": "ocem.p14t_adaptation_plan.v1",
        "status": "PASS",
        "dataset": "phoenix2014t",
        "target_data_policy": "train_manifest_only",
        "source_manifest": {
            "path": str(train_manifest.resolve()),
            "sha256": sha256_file(train_manifest),
            "sample_count": len(source_ids),
            "sample_ids_sha256": canonical_json_sha256(sorted(source_ids)),
        },
        "forbidden_manifests": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "sample_count": len(forbidden_by_manifest[str(path.resolve())]),
            }
            for path in forbidden_paths
        ],
        "forbidden_overlap_count": 0,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "expected_classifier_classes_from_vocabulary": len(vocabulary),
        },
        "class_vocabulary": {
            "path": str(class_vocabulary.resolve()),
            "sha256": sha256_file(class_vocabulary),
            "classes": len(vocabulary),
            "names_sha256": canonical_json_sha256(vocabulary),
        },
        "pseudo_label_recipe": {
            "confidence_threshold": 0.6,
            "clip_frames": 16,
            "window_stride": 1,
            "merge_start_distance_frames": 3,
            "suppress_start_distance_frames": 24,
            "selection": "per-class/per-video descending confidence",
            "source": "pinned CiCo epoch_pseudo.py; port required for manifest scoping",
        },
        "trainer_recipe": {
            "epochs": 15,
            "batch_size": 4,
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "encoder_lr_coefficient": 1.0,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "schedule_epochs": [20, 40],
            "schedule_gamma": 0.1,
            "augmentation": "CiCo train crop/scale/color-jitter/hflip",
        },
        "split_policy": {
            "version": "sha256_modulus_v1",
            "seed": seed,
            "holdout_modulus": holdout_modulus,
            "train_count": len(train_ids),
            "holdout_count": len(holdout_ids),
            "assignment_sha256": canonical_json_sha256(assignments),
        },
        "assignments": assignments,
        "validation_or_test_used": False,
        "ready_for_pseudo_label_generation": True,
        "ready_for_adaptation_training": False,
        "training_blocker": (
            "Implement and smoke-test the manifest-indexed pseudo-clip loader; the pinned "
            "upstream P14T trainer references missing/hard-coded inputs."
        ),
    }
