#!/usr/bin/env python3
"""Reproducible, annotation-only structure analysis for sentence-level SLRet.

This script does not decode videos, load model checkpoints, or score TEST with a
learned model.  It measures split composition, semantic/source grouping, and
lexical compositional support directly from the locally available annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PH_ROOT = Path("/home/dongvk/datasets/phoenix14T")
CSL_ROOT = Path("/home/dongvk/datasets/CSL_Daily_Sentence_Crop")
H2_ROOT = Path("/home/shared_data/sign_language/How2Sign")


@dataclass(frozen=True)
class Row:
    dataset: str
    split: str
    sample_id: str
    text: str
    semantic_group: str | None = None
    source_group: str | None = None
    signer: str | None = None
    duration: float | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def lexical_units(text: str) -> tuple[str, ...]:
    """Tokenize alphabetic languages by words and Chinese by characters."""
    normalized = normalize_text(text)
    units: list[str] = []
    word: list[str] = []

    def flush() -> None:
        if word:
            units.append("".join(word))
            word.clear()

    for char in normalized:
        if is_cjk(char):
            flush()
            units.append(char)
        elif char.isalnum() or (char in {"'", "-"} and word):
            word.append(char)
        else:
            flush()
    flush()
    return tuple(unit for unit in units if unit.strip("'-"))


def ngrams(tokens: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    return {tokens[i : i + n] for i in range(max(0, len(tokens) - n + 1))}


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "min": None, "p25": None,
                "p50": None, "p75": None, "p95": None, "max": None}

    def quantile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


def load_ph() -> tuple[dict[str, list[Row]], list[dict[str, object]]]:
    splits: dict[str, list[Row]] = {}
    sources: list[dict[str, object]] = []
    annotation_root = (
        PH_ROOT
        / "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual"
    )
    for split in ("train", "dev", "test"):
        path = annotation_root / f"PHOENIX-2014-T.{split}.corpus.csv"
        with path.open(encoding="utf-8") as stream:
            records = list(csv.DictReader(stream, delimiter="|"))
        splits[split] = [
            Row(
                dataset="ph",
                split=split,
                sample_id=record["name"],
                text=record["translation"],
                source_group=record["name"].rsplit("-", 1)[0],
                signer=record["speaker"],
            )
            for record in records
        ]
        sources.append({"path": str(path), "sha256": sha256(path), "rows": len(records)})
    return splits, sources


def load_csl() -> tuple[dict[str, list[Row]], list[dict[str, object]]]:
    splits: dict[str, list[Row]] = {}
    sources: list[dict[str, object]] = []
    for split in ("train", "dev", "test"):
        path = CSL_ROOT / f"{split}_data_with_num_frames.csv"
        with path.open(encoding="utf-8-sig") as stream:
            records = list(csv.DictReader(stream))
        rows = []
        for record in records:
            parts = record["name"].split("_")
            rows.append(
                Row(
                    dataset="csl",
                    split=split,
                    sample_id=record["name"],
                    text=record["text"],
                    semantic_group=parts[0],
                    signer=parts[1] if len(parts) > 1 else None,
                )
            )
        splits[split] = rows
        sources.append({"path": str(path), "sha256": sha256(path), "rows": len(rows)})
    return splits, sources


def load_h2() -> tuple[dict[str, list[Row]], list[dict[str, object]]]:
    splits: dict[str, list[Row]] = {}
    sources: list[dict[str, object]] = []

    train_path = H2_ROOT / "train/train_label/labels.train"
    with train_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    splits["train"] = [
        Row(
            dataset="h2",
            split="train",
            sample_id=record["sentence_name"],
            text=record["text"],
            source_group=record["video_id"],
            duration=float(record["duration"]),
        )
        for record in records
    ]
    sources.append({"path": str(train_path), "sha256": sha256(train_path), "rows": len(records)})

    dev_path = H2_ROOT / "eval/how2sign_realigned_val.csv"
    with dev_path.open(encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    splits["dev"] = [
        Row(
            dataset="h2",
            split="dev",
            sample_id=record["SENTENCE_NAME"],
            text=record["SENTENCE"],
            source_group=record["VIDEO_ID"],
            duration=float(record["END_REALIGNED"]) - float(record["START_REALIGNED"]),
        )
        for record in records
    ]
    sources.append({"path": str(dev_path), "sha256": sha256(dev_path), "rows": len(records)})
    return splits, sources


def split_summary(rows: list[Row]) -> dict[str, object]:
    texts = Counter(normalize_text(row.text) for row in rows)
    semantics = Counter(row.semantic_group for row in rows if row.semantic_group)
    sources = Counter(row.source_group for row in rows if row.source_group)
    signers = Counter(row.signer for row in rows if row.signer)
    tokenized = [lexical_units(row.text) for row in rows]
    return {
        "rows": len(rows),
        "unique_normalized_texts": len(texts),
        "rows_in_repeated_text_classes": sum(count for count in texts.values() if count > 1),
        "text_multiplicity": describe(texts.values()),
        "unique_semantic_groups": len(semantics),
        "rows_in_repeated_semantic_groups": sum(count for count in semantics.values() if count > 1),
        "semantic_group_multiplicity": describe(semantics.values()),
        "unique_source_groups": len(sources),
        "source_group_multiplicity": describe(sources.values()),
        "signer_counts": dict(sorted(signers.items())),
        "lexical_units_per_row": describe(len(tokens) for tokens in tokenized),
        "duration_seconds": describe(row.duration for row in rows if row.duration is not None),
    }


def overlap(left: list[Row], right: list[Row]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in ("text", "semantic_group", "source_group", "signer"):
        if field == "text":
            left_values = {normalize_text(row.text) for row in left}
            right_values = [normalize_text(row.text) for row in right]
        else:
            left_values = {getattr(row, field) for row in left if getattr(row, field)}
            right_values = [getattr(row, field) for row in right if getattr(row, field)]
        right_set = set(right_values)
        result[field] = {
            "left_unique": len(left_values),
            "right_unique": len(right_set),
            "intersection_unique": len(left_values & right_set),
            "right_rows_seen_in_left": sum(value in left_values for value in right_values),
            "right_rows_considered": len(right_values),
        }
    return result


def lexical_support(train: list[Row], evaluation: list[Row]) -> dict[str, object]:
    train_ngrams = {n: set() for n in (1, 2, 3)}
    for row in train:
        tokens = lexical_units(row.text)
        for n in train_ngrams:
            train_ngrams[n].update(ngrams(tokens, n))

    coverage: dict[int, list[float]] = {n: [] for n in train_ngrams}
    all_unigrams_seen: list[bool] = []
    all_bigrams_seen: list[bool] = []
    exact_train = {normalize_text(row.text) for row in train}
    exact_seen = []
    for row in evaluation:
        tokens = lexical_units(row.text)
        exact_seen.append(normalize_text(row.text) in exact_train)
        for n, reference in train_ngrams.items():
            row_ngrams = ngrams(tokens, n)
            fraction = (
                sum(item in reference for item in row_ngrams) / len(row_ngrams)
                if row_ngrams
                else 1.0
            )
            coverage[n].append(fraction)
        unigram_set = ngrams(tokens, 1)
        bigram_set = ngrams(tokens, 2)
        all_unigrams_seen.append(not unigram_set or unigram_set <= train_ngrams[1])
        all_bigrams_seen.append(not bigram_set or bigram_set <= train_ngrams[2])

    n_rows = len(evaluation)
    compositional = [
        unigram and not bigram and not exact
        for unigram, bigram, exact in zip(all_unigrams_seen, all_bigrams_seen, exact_seen)
    ]
    return {
        "reference_train_rows": len(train),
        "evaluation_rows": n_rows,
        "ngram_coverage_fraction": {
            str(n): describe(values) for n, values in coverage.items()
        },
        "rows_with_all_unigrams_seen": sum(all_unigrams_seen),
        "rows_with_all_bigrams_seen": sum(all_bigrams_seen),
        "rows_exact_text_seen": sum(exact_seen),
        "rows_compositional_by_definition": sum(compositional),
        "compositional_definition": (
            "all lexical unigrams occur in TRAIN, at least one adjacent bigram is unseen, "
            "and the complete normalized text is unseen"
        ),
    }


def analyze() -> dict[str, object]:
    loaders = {"ph": load_ph, "csl": load_csl, "h2": load_h2}
    datasets: dict[str, object] = {}
    annotations: list[dict[str, object]] = []
    for name, loader in loaders.items():
        splits, sources = loader()
        annotations.extend(sources)
        split_summaries = {split: split_summary(rows) for split, rows in splits.items()}
        cross_split = {}
        split_names = list(splits)
        for i, left_name in enumerate(split_names):
            for right_name in split_names[i + 1 :]:
                cross_split[f"{left_name}->{right_name}"] = overlap(
                    splits[left_name], splits[right_name]
                )
        support = {
            split: lexical_support(splits["train"], rows)
            for split, rows in splits.items()
            if split != "train"
        }
        datasets[name] = {
            "splits": split_summaries,
            "cross_split_overlap": cross_split,
            "lexical_support_against_train": support,
        }
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Annotation-only descriptive analysis. Group IDs are parsed from released names; "
            "they are not independently verified recording identities. No causal inference."
        ),
        "annotations": annotations,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
