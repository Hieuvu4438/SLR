from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slr_common.utils import atomic_json_dump, ordered_hash, sha256_file


@dataclass(frozen=True)
class TransferRecord:
    pair_id: str
    video_file: str
    text: str


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        magic = handle.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else Path.open
    if opener is gzip.open:
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _pickle_records(path: Path) -> list[TransferRecord]:
    value = _read_pickle(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected a dictionary annotation: {path}")
    records = []
    for key, item in value.items():
        if not isinstance(item, dict):
            raise ValueError(f"annotation row is not a mapping: {path}:{key}")
        video_file = str(item.get("video_path", item.get("video", key)))
        records.append(
            TransferRecord(
                pair_id=video_file,
                video_file=video_file,
                text=str(item.get("text", "")),
            )
        )
    return records


def _how2sign_train_records(path: Path) -> list[TransferRecord]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                video_file = str(value["video_path"])
                text = str(value["text"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid How2Sign JSONL row {path}:{line_number}") from error
            records.append(TransferRecord(video_file, video_file, text))
    return records


def _how2sign_dev_records(path: Path) -> list[TransferRecord]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"SENTENCE_NAME", "SENTENCE"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"How2Sign dev TSV lacks columns {sorted(required)}: {path}")
        for row in reader:
            video_file = f"{row['SENTENCE_NAME']}.mp4"
            records.append(TransferRecord(video_file, video_file, str(row["SENTENCE"])))
    return records


def _csl_records(path: Path) -> list[TransferRecord]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "text", "video_path"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"CSL-Daily CSV lacks columns {sorted(required)}: {path}")
        for row in reader:
            records.append(
                TransferRecord(
                    pair_id=str(row["name"]),
                    video_file=str(row["video_path"]),
                    text=str(row["text"]),
                )
            )
    return records


def _split_audit(
    records: list[TransferRecord],
    annotation: Path,
    video_root: Path,
    *,
    report_extra_files: bool,
) -> dict[str, Any]:
    pair_ids = [record.pair_id for record in records]
    video_names = [record.video_file for record in records]
    pair_duplicates = sorted(
        identifier for identifier, count in Counter(pair_ids).items() if count > 1
    )
    video_duplicates = sorted(
        identifier for identifier, count in Counter(video_names).items() if count > 1
    )
    available = {path.name for path in video_root.glob("*.mp4") if path.is_file()}
    required = set(video_names)
    missing = sorted(required - available)
    extras = sorted(available - required) if report_extra_files else []
    blank_text = [record.pair_id for record in records if not record.text.strip()]
    ready = (
        bool(records)
        and not pair_duplicates
        and not video_duplicates
        and not missing
        and not extras
        and not blank_text
    )
    return {
        "status": "ready" if ready else "blocked_assets",
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha256_file(annotation),
        "video_root": str(video_root.resolve()),
        "annotation_count": len(records),
        "unique_pair_count": len(set(pair_ids)),
        "unique_video_reference_count": len(required),
        "video_file_count": len(available),
        "ordered_pair_id_hash": ordered_hash(pair_ids),
        "video_filename_set_hash": ordered_hash(sorted(available)),
        "duplicate_pair_count": len(pair_duplicates),
        "duplicate_pair_preview": pair_duplicates[:20],
        "duplicate_video_reference_count": len(video_duplicates),
        "duplicate_video_reference_preview": video_duplicates[:20],
        "missing_video_count": len(missing),
        "missing_video_preview": missing[:20],
        "extra_video_count": len(extras) if report_extra_files else None,
        "extra_video_preview": extras[:20] if report_extra_files else [],
        "blank_text_count": len(blank_text),
        "blank_text_preview": blank_text[:20],
    }


def _cross_split_pair_overlap(
    records_by_split: dict[str, list[TransferRecord]],
) -> dict[str, dict[str, Any]]:
    result = {}
    splits = sorted(records_by_split)
    for index, left in enumerate(splits):
        left_ids = {record.pair_id for record in records_by_split[left]}
        for right in splits[index + 1 :]:
            shared = sorted(left_ids & {record.pair_id for record in records_by_split[right]})
            result[f"{left}__{right}"] = {
                "count": len(shared),
                "preview": shared[:20],
            }
    return result


def _finalize(
    dataset: str,
    records_by_split: dict[str, list[TransferRecord]],
    splits: dict[str, dict[str, Any]],
    root: Path,
    *,
    additional: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overlaps = _cross_split_pair_overlap(records_by_split)
    ready = all(value["status"] == "ready" for value in splits.values()) and not any(
        value["count"] for value in overlaps.values()
    )
    disk = shutil.disk_usage(root)
    return {
        "schema_version": 1,
        "dataset": dataset,
        "status": "ready_for_feature_extraction" if ready else "blocked_assets",
        "audit_scope": "identity_split_text_presence_and_video_file_existence_only",
        "test_content_used_for_tuning": False,
        "splits": splits,
        "cross_split_pair_overlap": overlaps,
        "disk": {
            "free_bytes": disk.free,
            "minimum_project_disk_reserve_bytes": 20 * 1024**3,
            "feature_extraction_size_estimate_included": False,
        },
        **(additional or {}),
    }


def audit_how2sign(
    root: Path, auxiliary_label_root: Path, *, subset_root: Path | None = None
) -> dict[str, Any]:
    annotations = {
        "train": root / "train" / "train_label" / "labels.train",
        "dev": root / "eval" / "how2sign_realigned_val.csv",
        "test": auxiliary_label_root / "labels.test",
    }
    video_roots = {
        "train": root / "train" / "raw_videos",
        "dev": root / "eval" / "raw_videos",
        "test": root / "test" / "raw_videos",
    }
    records_by_split = {
        "train": _how2sign_train_records(annotations["train"]),
        "dev": _how2sign_dev_records(annotations["dev"]),
        "test": _pickle_records(annotations["test"]),
    }
    splits = {
        split: _split_audit(
            records,
            annotations[split],
            video_roots[split],
            report_extra_files=True,
        )
        for split, records in records_by_split.items()
    }
    additional: dict[str, Any] = {
        "caption_language": "en",
        "view": "rgb_front_sentence_clips",
    }
    if subset_root is not None:
        subset_annotation = subset_root / "labels.train"
        subset_records = _pickle_records(subset_annotation)
        additional["pilot_subset"] = {
            **_split_audit(
                subset_records,
                subset_annotation,
                subset_root / "raw_videos",
                report_extra_files=True,
            ),
            "protocol_role": "nonstandard_train_only_engineering_subset_not_gate_x",
        }
    return _finalize("how2sign", records_by_split, splits, root, additional=additional)


def audit_csl_daily(root: Path) -> dict[str, Any]:
    annotations = {
        split: root / f"{split}_data_with_num_frames.csv" for split in ("train", "dev", "test")
    }
    records_by_split = {split: _csl_records(path) for split, path in annotations.items()}
    video_root = root / "videos"
    splits = {
        split: _split_audit(
            records,
            annotations[split],
            video_root,
            report_extra_files=False,
        )
        for split, records in records_by_split.items()
    }
    referenced = {record.video_file for records in records_by_split.values() for record in records}
    available = {path.name for path in video_root.glob("*.mp4") if path.is_file()}
    extras = sorted(available - referenced)
    union = {
        "referenced_video_count": len(referenced),
        "video_file_count": len(available),
        "extra_video_count": len(extras),
        "extra_video_preview": extras[:20],
    }
    result = _finalize(
        "csl_daily",
        records_by_split,
        splits,
        root,
        additional={"caption_language": "zh-CN", "union_video_inventory": union},
    )
    if extras:
        result["status"] = "blocked_assets"
    return result


def write_csl_video_lists(
    root: Path, output_root: Path, splits: list[str]
) -> dict[str, dict[str, Any]]:
    """Write exact, ordered flat-root video lists without copying dataset files."""

    if not splits or len(set(splits)) != len(splits):
        raise ValueError("video-list splits must be non-empty and unique")
    invalid = set(splits) - {"train", "dev", "test"}
    if invalid:
        raise ValueError(f"invalid video-list splits: {sorted(invalid)}")
    video_root = root / "videos"
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Any]] = {}
    for split in splits:
        annotation = root / f"{split}_data_with_num_frames.csv"
        records = _csl_records(annotation)
        names = [record.video_file for record in records]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate video references in split={split}")
        for name in names:
            path = Path(name)
            if (
                path.is_absolute()
                or len(path.parts) != 1
                or path.suffix.lower() != ".mp4"
                or not (video_root / path).is_file()
            ):
                raise ValueError(f"invalid or missing flat-root video for split={split}: {name}")
        output = output_root / f"{split}.txt"
        temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for name in names:
                handle.write(name + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
        result[split] = {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "annotation": str(annotation.resolve()),
            "annotation_sha256": sha256_file(annotation),
            "count": len(names),
            "ordered_video_filename_hash": ordered_hash(names),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit existing transfer-dataset annotations and video identities"
    )
    parser.add_argument("--dataset", choices=("how2sign", "csl_daily"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--auxiliary-label-root")
    parser.add_argument("--subset-root")
    parser.add_argument("--video-list-root")
    parser.add_argument(
        "--video-list-splits",
        nargs="+",
        choices=("train", "dev", "test"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.dataset == "how2sign":
        if args.auxiliary_label_root is None:
            parser.error("How2Sign audit requires --auxiliary-label-root")
        result = audit_how2sign(
            root,
            Path(args.auxiliary_label_root),
            subset_root=Path(args.subset_root) if args.subset_root else None,
        )
    else:
        result = audit_csl_daily(root)
    if bool(args.video_list_root) != bool(args.video_list_splits):
        parser.error("--video-list-root and --video-list-splits must be provided together")
    if args.video_list_root:
        if args.dataset != "csl_daily":
            parser.error("ordered video-list export is currently supported for CSL-Daily")
        if result["status"] != "ready_for_feature_extraction":
            raise ValueError("refusing to export video lists from a blocked asset audit")
        result["video_lists"] = write_csl_video_lists(
            root, Path(args.video_list_root), args.video_list_splits
        )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready_for_feature_extraction" else 2


if __name__ == "__main__":
    raise SystemExit(main())
