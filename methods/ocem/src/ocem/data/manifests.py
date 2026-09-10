"""Atomic JSONL manifest writing and ID audits."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Any

from ocem.provenance.hashes import sha256_file


class ManifestError(ValueError):
    """Raised when IDs or record counts violate a manifest contract."""


def audit_sample_ids(records: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:5])
        raise ManifestError(f"duplicate sample IDs: {preview}")


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    destination = Path(path)
    materialized = list(records)
    audit_sample_ids(materialized)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialized:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return {
        "path": str(destination.resolve()),
        "records": len(materialized),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }

