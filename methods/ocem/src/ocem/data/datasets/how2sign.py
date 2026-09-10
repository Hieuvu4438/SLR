"""How2Sign manifest construction with explicit mismatch accounting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ocem.data.datasets.phoenix import Probe, _probe_all
from ocem.data.manifests import write_jsonl
from ocem.data.schema import SampleRecord
from ocem.data.video import VideoMetadata, probe_video
from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class How2SignDataError(ValueError):
    """Raised when a realigned annotation snapshot is structurally invalid."""


_REQUIRED_COLUMNS = {
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START_REALIGNED",
    "END_REALIGNED",
    "SENTENCE",
}


def _read_annotation(path: Path) -> list[tuple[dict[str, str], str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise How2SignDataError(f"empty annotation file: {path}")
    header = next(csv.reader([lines[0]], delimiter="\t"))
    missing = sorted(_REQUIRED_COLUMNS - set(header))
    if missing:
        raise How2SignDataError(f"missing columns in {path}: {', '.join(missing)}")
    rows = []
    for line_number, raw_line in enumerate(lines[1:], start=2):
        values = next(csv.reader([raw_line], delimiter="\t"))
        if len(values) != len(header):
            raise How2SignDataError(f"{path}:{line_number}: expected {len(header)} columns")
        rows.append((dict(zip(header, values, strict=True)), raw_line))
    return rows


def prepare_split(
    *,
    split: str,
    annotation_path: str | Path,
    video_dir: str | Path,
    raw_relprefix: str,
    output_dir: str | Path,
    annotation_count: int,
    published_cico_count: int,
    workers: int = 8,
    probe: Probe = probe_video,
) -> dict[str, Any]:
    annotation = Path(annotation_path).resolve()
    videos = Path(video_dir).resolve()
    rows = _read_annotation(annotation)
    sample_ids = [row[0]["SENTENCE_NAME"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise How2SignDataError(f"{split}: duplicate SENTENCE_NAME values")
    if len(rows) != annotation_count:
        raise How2SignDataError(f"{split}: expected {annotation_count} annotations, found {len(rows)}")
    video_paths = {sample_id: videos / f"{sample_id}.mp4" for sample_id in sample_ids}
    existing_paths = {sample_id: path for sample_id, path in video_paths.items() if path.is_file()}
    metadata = _probe_all(existing_paths, probe, workers)
    records = []
    exclusions = []
    raw_index = []
    for row, raw_line in rows:
        sample_id = row["SENTENCE_NAME"]
        video_path = video_paths[sample_id]
        observed = metadata.get(sample_id)
        reason = None
        try:
            source_start = float(row["START_REALIGNED"])
            source_end = float(row["END_REALIGNED"])
            if source_start < 0 or source_end <= source_start:
                reason = "invalid_annotation_interval"
        except ValueError:
            source_start = source_end = 0.0
            reason = "invalid_annotation_interval"
        if not video_path.is_file():
            reason = "missing_named_video"
        elif isinstance(observed, Exception):
            reason = f"video_probe_failed:{type(observed).__name__}:{observed}"
        if reason is None:
            assert isinstance(observed, VideoMetadata)
            fps_num, fps_den = observed.fps_num, observed.fps_den
            frame_count, end_time = observed.frame_count, observed.duration_s
            decode_status = "verified_metadata"
            raw_bytes = video_path.stat().st_size
            raw_index.append(
                {"sample_id": sample_id, "relpath": video_path.name, "bytes": raw_bytes}
            )
        else:
            fps_num = fps_den = 0
            frame_count = None
            end_time = 0.0
            decode_status = "failed"
            raw_bytes = video_path.stat().st_size if video_path.is_file() else None
        caption = row["SENTENCE"]
        record = SampleRecord(
            schema_version="ocem.sample.v1",
            dataset="how2sign",
            split=split,
            sample_id=sample_id,
            video_id=row["VIDEO_ID"],
            source_group_id=row["VIDEO_ID"],
            signer_id=None,
            raw_relpath=f"{raw_relprefix}/{video_path.name}",
            caption_raw=caption,
            caption_sha256=hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            annotation_sha256=hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
            start_time_s=0.0,
            end_time_s=end_time,
            fps_num=fps_num,
            fps_den=fps_den,
            decode_status=decode_status,
            include_in_protocol=reason is None,
            exclusion_reason=reason,
            frame_count=frame_count,
            raw_bytes=raw_bytes,
            source_annotation={
                "sentence_id": row["SENTENCE_ID"],
                "video_name": row["VIDEO_NAME"],
                "start_realigned_s": source_start,
                "end_realigned_s": source_end,
            },
        ).to_dict()
        records.append(record)
        if reason is not None:
            exclusions.append({"sample_id": sample_id, "reason": reason})
    output = Path(output_dir)
    original = write_jsonl(output / f"original_{split}.jsonl", records)
    usable = write_jsonl(output / f"{split}.jsonl", [r for r in records if r["include_in_protocol"]])
    excluded = write_jsonl(output / f"exclusions_{split}.jsonl", exclusions)
    # Matching a published count is insufficient evidence that the same IDs were used.
    # H2S stays unverified until an explicit published-ID lock is supplied and compared.
    equivalence = "UNVERIFIED"
    return {
        "split": split,
        "annotation_count": annotation_count,
        "annotation_rows": len(rows),
        "published_cico_count": published_cico_count,
        "usable_count": usable["records"],
        "excluded_count": excluded["records"],
        "annotation_file_sha256": sha256_file(annotation),
        "raw_index_basis": "sample_id+relative_filename+bytes; individual video content hashes not computed",
        "raw_index_sha256": canonical_json_sha256(raw_index),
        "manifests": {"original": original, "usable": usable, "exclusions": excluded},
        "protocol_equivalence": equivalence,
        "status": "BLOCKED_RESOURCE",
    }


def prepare_how2sign(
    protocol: Mapping[str, Any], output_dir: str | Path, *, workers: int = 8, probe: Probe = probe_video
) -> dict[str, Any]:
    split_reports = {}
    for split, settings in protocol["official_splits"].items():
        split_reports[split] = prepare_split(
            split=split,
            annotation_path=settings["annotation"],
            video_dir=settings["video_dir"],
            raw_relprefix=settings["raw_relprefix"],
            output_dir=output_dir,
            annotation_count=int(settings["annotation_count"]),
            published_cico_count=int(settings["published_cico_count"]),
            workers=workers,
            probe=probe,
        )
    status = "PASS" if all(report["status"] == "PASS" for report in split_reports.values()) else "BLOCKED_RESOURCE"
    body = {
        "schema_version": "ocem.protocol_lock.v1",
        "dataset": "how2sign",
        "status": status,
        "protocol_equivalence": "PASS" if status == "PASS" else "UNVERIFIED",
        "split_reports": split_reports,
        "evaluation": protocol["evaluation"],
        "galleries": protocol["galleries"],
        "preprocessing": protocol["preprocessing"],
        "baseline": protocol["baseline"],
        "feature_lock_sha256": None,
        "baseline_reproduction_status": "NOT_RUN",
    }
    body["lock_sha256"] = canonical_json_sha256(body)
    destination = Path(output_dir) / "protocol_lock.json"
    destination.write_text(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body
