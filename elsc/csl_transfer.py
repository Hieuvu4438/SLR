from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Callable

import torch

from elsc.utils import atomic_json_dump, ordered_hash, sha256_file


_VIDEO_ID = re.compile(r"^(S\d{6})_P\d{4}_T\d{2}$")


def _rows(root: Path, split: str) -> tuple[Path, list[dict[str, str]]]:
    if split not in {"train", "dev"}:
        raise ValueError("CSL transfer preparation is restricted to train and dev")
    path = root / f"{split}_data_with_num_frames.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"name", "text", "video_path"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"CSL-Daily {split} CSV lacks {sorted(required)}: {path}")
    return path, rows


def _caption_id(video_id: str) -> str:
    match = _VIDEO_ID.fullmatch(video_id)
    if match is None:
        raise ValueError(f"unexpected CSL-Daily video ID: {video_id}")
    return match.group(1)


def dev_caption_groups(root: Path) -> tuple[Path, list[tuple[str, str]]]:
    path, rows = _rows(root, "dev")
    groups: dict[str, str] = {}
    for row in rows:
        caption_id = _caption_id(row["name"])
        text = row["text"].strip()
        if not text:
            raise ValueError(f"blank dev caption: {row['name']}")
        previous = groups.setdefault(caption_id, text)
        if previous != text:
            raise ValueError(f"inconsistent dev caption group: {caption_id}")
    return path, list(groups.items())


def translate_dev(
    root: Path,
    translate: Callable[[list[str]], list[str]],
    *,
    model: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    annotation, groups = dev_caption_groups(root)
    translated = translate([text for _, text in groups])
    if len(translated) != len(groups):
        raise ValueError("translator returned the wrong number of captions")
    entries = []
    for (caption_id, source_text), target_text in zip(groups, translated, strict=True):
        target_text = str(target_text).strip()
        if not target_text:
            raise ValueError(f"translator returned blank text for {caption_id}")
        entries.append(
            {
                "caption_id": caption_id,
                "source_language": "zh-CN",
                "source_text": source_text,
                "target_language": "en",
                "target_text": target_text,
            }
        )
    return {
        "schema_version": 1,
        "dataset": "csl_daily",
        "split": "dev",
        "source_annotation": str(annotation.resolve()),
        "source_annotation_sha256": sha256_file(annotation),
        "translation_role": "fixed_dev_input_for_english_cico_baseline_and_method",
        "model": model,
        "generation": generation,
        "caption_group_count": len(entries),
        "ordered_caption_id_hash": ordered_hash(item["caption_id"] for item in entries),
        "test_annotation_accessed": False,
        "translations": entries,
    }


def _atomic_pickle(value: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def build_annotations(
    root: Path,
    upstream_train: Path,
    dev_translation_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    with upstream_train.open("rb") as handle:
        upstream = pickle.load(handle)
    if not isinstance(upstream, dict):
        raise ValueError("upstream CSL train annotation must be a dictionary")
    flattened: dict[str, tuple[str, dict[str, Any]]] = {}
    for caption_id, items in upstream.items():
        if not isinstance(items, list) or not items:
            raise ValueError(f"invalid upstream caption group: {caption_id}")
        for item in items:
            video_id = str(item.get("video_name", ""))
            if not video_id or video_id in flattened:
                raise ValueError(f"invalid or duplicate upstream video ID: {video_id}")
            flattened[video_id] = (str(caption_id), item)

    train_csv, train_rows = _rows(root, "train")
    train: dict[str, dict[str, str]] = {}
    for row in train_rows:
        video_id = row["name"]
        if video_id not in flattened:
            raise ValueError(f"train video absent from pinned upstream annotation: {video_id}")
        caption_id, item = flattened[video_id]
        if caption_id != _caption_id(video_id):
            raise ValueError(f"upstream caption group mismatch: {video_id}")
        if str(item.get("ori_text", "")) != row["text"]:
            raise ValueError(f"upstream Chinese caption mismatch: {video_id}")
        target = str(item.get("text", "")).strip()
        if not target:
            raise ValueError(f"blank upstream English caption: {video_id}")
        train[video_id] = {
            "video_name": video_id,
            "caption_id": caption_id,
            "ori_text": row["text"],
            "text": target,
        }
    if set(flattened) != set(train):
        raise ValueError("pinned upstream train annotation has videos outside official train CSV")

    translations = json.loads(dev_translation_path.read_text(encoding="utf-8"))
    if (
        translations.get("schema_version") != 1
        or translations.get("dataset") != "csl_daily"
        or translations.get("split") != "dev"
        or translations.get("test_annotation_accessed") is not False
    ):
        raise ValueError("invalid or test-contaminated dev translation artifact")
    dev_map = {
        item["caption_id"]: item
        for item in translations.get("translations", [])
    }
    if len(dev_map) != len(translations.get("translations", [])):
        raise ValueError("duplicate caption IDs in dev translation artifact")
    dev_csv, dev_rows = _rows(root, "dev")
    dev: dict[str, dict[str, str]] = {}
    for row in dev_rows:
        video_id = row["name"]
        caption_id = _caption_id(video_id)
        item = dev_map.get(caption_id)
        if item is None or item.get("source_text") != row["text"]:
            raise ValueError(f"missing or mismatched dev translation: {video_id}")
        target = str(item.get("target_text", "")).strip()
        if not target:
            raise ValueError(f"blank dev English caption: {video_id}")
        dev[video_id] = {
            "video_name": video_id,
            "caption_id": caption_id,
            "ori_text": row["text"],
            "text": target,
        }
    if set(dev_map) != {_caption_id(video_id) for video_id in dev}:
        raise ValueError("dev translation artifact contains captions outside official dev CSV")

    outputs = {"train": output_root / "train.pkl", "dev": output_root / "dev.pkl"}
    _atomic_pickle(train, outputs["train"])
    _atomic_pickle(dev, outputs["dev"])
    return {
        "schema_version": 1,
        "dataset": "csl_daily",
        "splits": {
            "train": {
                "count": len(train),
                "caption_group_count": len({item["caption_id"] for item in train.values()}),
                "official_annotation": str(train_csv.resolve()),
                "official_annotation_sha256": sha256_file(train_csv),
                "model_annotation": str(outputs["train"].resolve()),
                "model_annotation_sha256": sha256_file(outputs["train"]),
                "translation_source": str(upstream_train.resolve()),
                "translation_source_sha256": sha256_file(upstream_train),
            },
            "dev": {
                "count": len(dev),
                "caption_group_count": len(dev_map),
                "official_annotation": str(dev_csv.resolve()),
                "official_annotation_sha256": sha256_file(dev_csv),
                "model_annotation": str(outputs["dev"].resolve()),
                "model_annotation_sha256": sha256_file(outputs["dev"]),
                "translation_source": str(dev_translation_path.resolve()),
                "translation_source_sha256": sha256_file(dev_translation_path),
            },
        },
        "test_annotation_accessed": False,
    }


def _load_transformer(model_root: Path, device: torch.device, batch_size: int):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_root, local_files_only=True).to(device)
    model.eval().requires_grad_(False)

    def run(texts: list[str]) -> list[str]:
        result: list[str] = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(**encoded, num_beams=4, max_new_tokens=128)
            result.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        return result

    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare test-isolated CSL-Daily transfer inputs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    translate_parser = subparsers.add_parser("translate-dev")
    translate_parser.add_argument("--csl-root", required=True)
    translate_parser.add_argument("--model-root", required=True)
    translate_parser.add_argument("--model-id", required=True)
    translate_parser.add_argument("--model-revision", required=True)
    translate_parser.add_argument("--device", default="cpu")
    translate_parser.add_argument("--batch-size", type=int, default=32)
    translate_parser.add_argument("--output", required=True)
    build_parser = subparsers.add_parser("build-annotations")
    build_parser.add_argument("--csl-root", required=True)
    build_parser.add_argument("--upstream-train", required=True)
    build_parser.add_argument("--dev-translations", required=True)
    build_parser.add_argument("--output-root", required=True)
    build_parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    if args.command == "translate-dev":
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        model_root = Path(args.model_root)
        model_files = {
            path.name: sha256_file(path)
            for path in sorted(model_root.iterdir())
            if path.is_file()
        }
        translator = _load_transformer(
            model_root, torch.device(args.device), args.batch_size
        )
        result = translate_dev(
            Path(args.csl_root),
            translator,
            model={
                "id": args.model_id,
                "revision": args.model_revision,
                "root": str(model_root.resolve()),
                "file_sha256": model_files,
            },
            generation={
                "num_beams": 4,
                "max_source_tokens": 128,
                "max_new_tokens": 128,
                "batch_size": args.batch_size,
                "device": args.device,
            },
        )
        atomic_json_dump(result, args.output)
    else:
        result = build_annotations(
            Path(args.csl_root),
            Path(args.upstream_train),
            Path(args.dev_translations),
            Path(args.output_root),
        )
        atomic_json_dump(result, args.report)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
