"""Manifest-scoped wrapper around the shared deterministic I3D extractor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class ExtractionRunError(ValueError):
    """Raised when adapted extraction cannot preserve its provenance contract."""


SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}


def _manifest_ids_and_names(path: Path, video_dir: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExtractionRunError(f"cannot read manifest {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExtractionRunError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(record, Mapping) or not record.get("sample_id"):
            raise ExtractionRunError(f"missing sample_id at {path}:{line_number}")
        sample_id = str(record["sample_id"])
        raw_relpath = Path(str(record.get("raw_relpath", "")))
        name = raw_relpath.name
        if (
            sample_id in seen
            or not name
            or Path(name).stem != sample_id
            or Path(name).suffix != ".mp4"
        ):
            raise ExtractionRunError(f"duplicate or inconsistent sample at {path}:{line_number}")
        if not (video_dir / name).is_file():
            raise ExtractionRunError(f"manifest video is missing: {video_dir / name}")
        seen.add(sample_id)
        ids.append(sample_id)
        names.append(name)
    if not ids:
        raise ExtractionRunError(f"manifest is empty: {path}")
    actual_ids = {item.stem for item in video_dir.glob("*.mp4")}
    if actual_ids != set(ids):
        raise ExtractionRunError(
            f"video directory IDs differ from manifest {path} "
            f"(missing={len(set(ids) - actual_ids)}, extra={len(actual_ids - set(ids))})"
        )
    return ids, names


def _write_list(names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{name}\n" for name in names)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_report(path: Path) -> Mapping[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionRunError(f"invalid shared extractor report {path}: {error}") from error
    if not isinstance(report, Mapping):
        raise ExtractionRunError(f"shared extractor report is not an object: {path}")
    return report


def extract_p14t_i3d_features(
    *,
    manifest_dir: str | Path,
    video_root: str | Path,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    output_root: str | Path,
    temporal_root: str | Path,
    shared_root: str | Path,
    extractor: str | Path,
    expected_extractor_sha256: str,
    upstream_i3d: str | Path,
    expected_upstream_i3d_sha256: str,
    splits: Sequence[str] = ("train", "validation", "test"),
    device: str = "cuda:0",
    batch_size: int = 32,
    min_free_disk_gib: float = 20.0,
    min_free_gpu_gib: float = 12.0,
    report_interval: int = 25,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract one stream while constraining every split to the OCEM manifest IDs."""

    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(split not in SPLIT_MAP for split in splits)
    ):
        raise ExtractionRunError(f"invalid or duplicate protocol splits: {list(splits)}")
    if batch_size < 1 or report_interval < 1:
        raise ExtractionRunError("batch_size and report_interval must be positive")
    manifest_dir, video_root = Path(manifest_dir), Path(video_root)
    checkpoint, output_root, temporal_root = (
        Path(checkpoint),
        Path(output_root),
        Path(temporal_root),
    )
    shared_root, extractor, upstream_i3d = Path(shared_root), Path(extractor), Path(upstream_i3d)
    if sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise ExtractionRunError("adapted checkpoint SHA-256 mismatch")
    if sha256_file(extractor) != expected_extractor_sha256:
        raise ExtractionRunError("shared extractor SHA-256 mismatch")
    if sha256_file(upstream_i3d) != expected_upstream_i3d_sha256:
        raise ExtractionRunError("upstream I3D implementation SHA-256 mismatch")
    if not (shared_root / "slr_common").is_dir():
        raise ExtractionRunError(f"shared slr_common package is missing: {shared_root}")

    manifests: dict[str, Any] = {}
    commands: list[list[str]] = []
    reports: dict[str, Any] = {}
    list_root = output_root / "_manifest_video_lists"
    for protocol_split in splits:
        cache_split = SPLIT_MAP[protocol_split]
        manifest_path = manifest_dir / f"{protocol_split}.jsonl"
        split_video_root = video_root / cache_split
        sample_ids, names = _manifest_ids_and_names(manifest_path, split_video_root)
        list_path = list_root / f"{cache_split}.txt"
        _write_list(names, list_path)
        manifests[protocol_split] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "samples": len(sample_ids),
            "sample_ids_sha256": canonical_json_sha256(sorted(sample_ids)),
            "video_list": str(list_path.resolve()),
            "video_list_sha256": sha256_file(list_path),
        }
        command = [
            sys.executable,
            str(extractor.resolve()),
            "--video-root",
            str(split_video_root.resolve()),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--expected-checkpoint-sha256",
            expected_checkpoint_sha256,
            "--stream-name",
            "domain_adapted_p14t",
            "--output-root",
            str(output_root.resolve()),
            "--temporal-metadata-root",
            str(temporal_root.resolve()),
            "--splits",
            cache_split,
            "--split-video-list",
            f"{cache_split}={list_path.resolve()}",
            "--stride",
            "1",
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--min-free-disk-gib",
            str(min_free_disk_gib),
            "--min-free-gpu-gib",
            str(min_free_gpu_gib),
            "--report-interval",
            str(report_interval),
            "--upstream-i3d",
            str(upstream_i3d.resolve()),
        ]
        if dry_run:
            command.append("--dry-run")
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(shared_root.resolve()) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        completed = subprocess.run(command, env=environment, check=False)
        commands.append(command)
        if completed.returncode != 0:
            raise ExtractionRunError(
                f"shared extractor failed for {protocol_split} with exit {completed.returncode}"
            )
        report_path = output_root / f"extraction_report_{cache_split}.json"
        report = _read_report(report_path)
        expected_status = "planned" if dry_run else "complete"
        if report.get("status") != expected_status:
            raise ExtractionRunError(
                f"shared extractor report for {protocol_split} has status {report.get('status')!r}"
            )
        if report.get("videos") != len(sample_ids) or report.get("checkpoint_sha256") != (
            expected_checkpoint_sha256
        ):
            raise ExtractionRunError(f"shared extractor report provenance mismatch: {report_path}")
        reports[protocol_split] = {
            "path": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
            "status": report["status"],
            "videos": report["videos"],
            "windows": report["windows"],
            "completed": report["completed"],
            "resumed": report["resumed"],
        }

    return {
        "schema_version": "ocem.p14t_adapted_extraction.v1",
        "status": "PLANNED" if dry_run else "PASS",
        "stream": "domain_adapted_p14t",
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": expected_checkpoint_sha256},
        "shared_extractor": {
            "path": str(extractor.resolve()),
            "sha256": expected_extractor_sha256,
        },
        "upstream_i3d": {
            "path": str(upstream_i3d.resolve()),
            "sha256": expected_upstream_i3d_sha256,
        },
        "manifests": manifests,
        "reports": reports,
        "commands": commands,
        "validation_or_test_used_for_training": False,
        "ready_for_feature_audit": not dry_run,
    }
