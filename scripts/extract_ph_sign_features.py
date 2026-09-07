from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from slr_common.resources import require_storage_budget
from slr_common.utils import atomic_json_dump, ordered_hash, sha256_file


ARCHIVE_SHA256 = "9ba1956cf416df9a31ae3d1a71a3fa9a2d1e2b3724670288b608c8d4eb895c51"
STREAMS = ("ph_domain_agnostic", "ph_domain_aware")
EXPECTED_PER_STREAM = 642


def extract_ph(archive: Path, output: Path, *, min_remaining_gib: float) -> dict:
    digest = sha256_file(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: {digest} != {ARCHIVE_SHA256}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    selected: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or len(path.parts) != 4 or path.parts[0] != "sign_features":
                continue
            if path.parts[1] not in STREAMS or path.parts[2] != "test" or path.suffix != ".pkl":
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {info.filename}")
            selected.append(info)
        counts = Counter(PurePosixPath(info.filename).parts[1] for info in selected)
        if counts != Counter({stream: EXPECTED_PER_STREAM for stream in STREAMS}):
            raise ValueError(f"unexpected PH archive counts: {dict(counts)}")
        selected_bytes = sum(info.file_size for info in selected)
        storage = require_storage_budget(
            output.parent,
            planned_write_bytes=selected_bytes,
            min_remaining_gib=min_remaining_gib,
            operation="selective PH release feature extraction",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        try:
            for info in selected:
                path = PurePosixPath(info.filename)
                destination = temporary.joinpath(*path.parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            files = sorted(
                path
                for stream in STREAMS
                for path in (temporary / stream / "test").glob("*.pkl")
            )
            if len(files) != EXPECTED_PER_STREAM * len(STREAMS):
                raise ValueError("selective extraction produced an unexpected file count")
            provenance = {
                "schema_version": 1,
                "artifact": "official_cico_sign_features_release_test_only",
                "archive": str(archive.resolve()),
                "archive_sha256": digest,
                "archive_source": (
                    "https://drive.google.com/file/d/"
                    "1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy/view"
                ),
                "archive_bytes": archive.stat().st_size,
                "selected_uncompressed_bytes": selected_bytes,
                "streams": {stream: {"test": counts[stream]} for stream in STREAMS},
                "contains_train": False,
                "contains_dev": False,
                "ordered_relative_path_hash": ordered_hash(
                    path.relative_to(temporary).as_posix() for path in files
                ),
                "storage_before_extraction": storage,
            }
            atomic_json_dump(provenance, temporary / "release_provenance.json")
            temporary.replace(output)
            return provenance
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely extract only Phoenix test features from the CiCo release archive"
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-free-disk-gib", type=float, default=20.0)
    args = parser.parse_args(argv)
    result = extract_ph(
        Path(args.archive), Path(args.output), min_remaining_gib=args.min_free_disk_gib
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
