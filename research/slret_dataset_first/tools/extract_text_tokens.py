#!/usr/bin/env python3
"""Cache contextual multilingual text tokens for the CoSign-LI matched pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
ANNOTATION_ROOT = Path(
    "/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/"
    "PHOENIX-2014-T/annotations/manual"
)
DEFAULT_SIGNREP = ROOT / "artifacts/slret_goal_v2/signrep-native-window-pilot-001"
DEFAULT_OUTPUT = ROOT / "artifacts/slret_goal/dataset-first-text-e5-001"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotations(split: str) -> tuple[Path, dict[str, str]]:
    path = ANNOTATION_ROOT / f"PHOENIX-2014-T.{split}.corpus.csv"
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    return path, {row["name"]: row["translation"] for row in rows}


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signrep-root", type=Path, default=DEFAULT_SIGNREP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=dtype
    ).eval().to(args.device)

    started = time.time()
    manifest: list[dict[str, object]] = []
    annotation_records = []
    with torch.inference_mode():
        for split in ("train", "dev"):
            annotation_path, captions = annotations(split)
            annotation_records.append(
                {"path": str(annotation_path), "sha256": sha256(annotation_path)}
            )
            ids = sorted(path.stem for path in (args.signrep_root / split).glob("*.npz"))
            missing = [sample_id for sample_id in ids if sample_id not in captions]
            if missing:
                raise KeyError(f"{split}: {len(missing)} SignRep IDs absent from annotations")
            for offset in range(0, len(ids), args.batch_size):
                batch_ids = ids[offset : offset + args.batch_size]
                texts = [captions[sample_id] for sample_id in batch_ids]
                raw_lengths = [
                    len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
                    for text in texts
                ]
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                special = encoded.pop("special_tokens_mask").bool()
                device_inputs = {key: value.to(args.device) for key, value in encoded.items()}
                hidden = model(**device_inputs).last_hidden_state.float().cpu()
                attention = encoded["attention_mask"].bool()
                content_mask = attention & ~special
                for index, sample_id in enumerate(batch_ids):
                    keep = content_mask[index]
                    if not keep.any():
                        raise ValueError(f"{sample_id}: no non-special text tokens")
                    output = args.output / split / f"{sample_id}.npz"
                    atomic_npz(
                        output,
                        tokens=hidden[index, keep].numpy().astype(np.float16),
                        input_ids=encoded["input_ids"][index, keep].numpy().astype(np.int32),
                    )
                    manifest.append(
                        {
                            "id": sample_id,
                            "split": split,
                            "tokens": int(keep.sum()),
                            "raw_tokens_with_special": raw_lengths[index],
                            "truncated": raw_lengths[index] > args.max_length,
                            "path": str(output.relative_to(ROOT)),
                        }
                    )
                print(
                    json.dumps(
                        {"event": "batch", "split": split, "completed": min(offset + len(batch_ids), len(ids)), "total": len(ids)}
                    ),
                    flush=True,
                )

    config = model.config
    report = {
        "run_id": args.output.name,
        "status": "complete",
        "model": args.model,
        "model_commit": getattr(config, "_commit_hash", None),
        "hidden_size": int(config.hidden_size),
        "dtype": str(dtype),
        "max_length": args.max_length,
        "signrep_root": str(args.signrep_root),
        "annotations": annotation_records,
        "rows": len(manifest),
        "truncated_rows": sum(bool(row["truncated"]) for row in manifest),
        "wall_seconds": time.time() - started,
        "test_used": False,
        "note": "Raw caption, no E5 instruction prefix; non-special contextual tokens only.",
    }
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
