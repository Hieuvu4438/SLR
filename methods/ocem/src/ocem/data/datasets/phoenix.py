"""PHOENIX-2014T manifest construction from official IDs and local MP4 clips."""

from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from ocem.data.manifests import write_jsonl
from ocem.data.schema import SampleRecord
from ocem.data.video import VideoMetadata, probe_video
from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class PhoenixDataError(ValueError):
    """Raised when official annotations cannot be parsed without ambiguity."""


Probe = Callable[[str | Path], VideoMetadata]
_EXPECTED_COLUMNS = ["name", "video", "start", "end", "speaker", "orth", "translation"]


def _read_annotation(path: Path) -> list[tuple[dict[str, str], str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise PhoenixDataError(f"empty annotation file: {path}")
    header = next(csv.reader([lines[0]], delimiter="|"))
    if header != _EXPECTED_COLUMNS:
        raise PhoenixDataError(f"unexpected columns in {path}: {header}")
    rows = []
    for line_number, raw_line in enumerate(lines[1:], start=2):
        values = next(csv.reader([raw_line], delimiter="|"))
        if len(values) != len(header):
            raise PhoenixDataError(f"{path}:{line_number}: expected {len(header)} columns")
        rows.append((dict(zip(header, values, strict=True)), raw_line))
    return rows


def _probe_all(
    paths: Mapping[str, Path], probe: Probe, workers: int
) -> dict[str, VideoMetadata | Exception]:
    if workers <= 1:
        answer = {}
        for sample_id, path in paths.items():
            try:
                answer[sample_id] = probe(path)
            except Exception as error:
                answer[sample_id] = error
        return answer
    answer = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(probe, path): sample_id for sample_id, path in paths.items()}
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                answer[sample_id] = future.result()
            except Exception as error:
                answer[sample_id] = error
    return answer


def prepare_split(
    *,
    split: str,
    annotation_path: str | Path,
    video_dir: str | Path,
    output_dir: str | Path,
    expected_count: int,
    workers: int = 8,
    probe: Probe = probe_video,
) -> dict[str, Any]:
    annotation = Path(annotation_path).resolve()
    videos = Path(video_dir).resolve()
    rows = _read_annotation(annotation)
    sample_ids = [row[0]["name"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise PhoenixDataError(f"{split}: duplicate official annotation IDs")
    if len(rows) != expected_count:
        raise PhoenixDataError(f"{split}: expected {expected_count} annotations, found {len(rows)}")
    video_paths = {sample_id: videos / f"{sample_id}.mp4" for sample_id in sample_ids}
    metadata = _probe_all(video_paths, probe, workers)
    records = []
    exclusions = []
    raw_index = []
    for (row, raw_line), sample_id in zip(rows, sample_ids, strict=True):
        video_path = video_paths[sample_id]
        observed = metadata[sample_id]
        reason = None
        if not video_path.is_file():
            reason = "missing_video"
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
        caption = row["translation"]
        record = SampleRecord(
            schema_version="ocem.sample.v1",
            dataset="phoenix2014t",
            split=split,
            sample_id=sample_id,
            video_id=sample_id,
            source_group_id=None,
            signer_id=row["speaker"] or None,
            raw_relpath=f"videos_phoenix/videos/{'dev' if split == 'validation' else split}/{video_path.name}",
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
                "video": row["video"],
                "start": row["start"],
                "end": row["end"],
                "orth": row["orth"],
            },
        ).to_dict()
        records.append(record)
        if reason is not None:
            exclusions.append({"sample_id": sample_id, "reason": reason})
    output = Path(output_dir)
    original = write_jsonl(output / f"original_{split}.jsonl", records)
    usable = write_jsonl(output / f"{split}.jsonl", [r for r in records if r["include_in_protocol"]])
    excluded = write_jsonl(output / f"exclusions_{split}.jsonl", exclusions)
    return {
        "split": split,
        "expected_count": expected_count,
        "annotation_rows": len(rows),
        "usable_count": usable["records"],
        "excluded_count": excluded["records"],
        "annotation_file_sha256": sha256_file(annotation),
        "raw_index_basis": "sample_id+relative_filename+bytes; individual video content hashes not computed",
        "raw_index_sha256": canonical_json_sha256(raw_index),
        "manifests": {"original": original, "usable": usable, "exclusions": excluded},
        "status": "PASS" if usable["records"] == expected_count and not exclusions else "BLOCKED_RESOURCE",
    }


def prepare_phoenix(
    protocol: Mapping[str, Any], output_dir: str | Path, *, workers: int = 8, probe: Probe = probe_video
) -> dict[str, Any]:
    split_reports = {}
    for split, settings in protocol["official_splits"].items():
        split_reports[split] = prepare_split(
            split=split,
            annotation_path=settings["annotation"],
            video_dir=settings["video_dir"],
            output_dir=output_dir,
            expected_count=int(settings["expected_count"]),
            workers=workers,
            probe=probe,
        )
    status = "PASS" if all(report["status"] == "PASS" for report in split_reports.values()) else "BLOCKED_RESOURCE"
    body = {
        "schema_version": "ocem.protocol_lock.v1",
        "dataset": "phoenix2014t",
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
