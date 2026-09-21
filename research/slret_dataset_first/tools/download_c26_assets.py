"""Download only the public C26 pose checkpoint and mT5 initialization files.

Network/disk job only: no GPU is touched. Run detached with stdout/stderr sent to
the job's run.log. The worker owns status.json and summary.json lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


RUN_ID = "c26-assets-001"
UNISIGN_REPO = "ZechengLi19/Uni-Sign"
UNISIGN_REV = "eab251b7fe7e8521afc0e67be98add670ea40a0d"
MT5_REPO = "google/mt5-base"
MT5_REV = "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f"
ASSETS = (
    (UNISIGN_REPO, UNISIGN_REV, "csl_daily_pose_only_slt.pth", "unisign"),
    (MT5_REPO, MT5_REV, "config.json", "mt5-base"),
    (MT5_REPO, MT5_REV, "generation_config.json", "mt5-base"),
    (MT5_REPO, MT5_REV, "pytorch_model.bin", "mt5-base"),
    (MT5_REPO, MT5_REV, "spiece.model", "mt5-base"),
    (MT5_REPO, MT5_REV, "tokenizer_config.json", "mt5-base"),
    (MT5_REPO, MT5_REV, "special_tokens_map.json", "mt5-base"),
)
TIMEOUT_SECONDS = 7200
FREE_SPACE_FLOOR = 20 * 1024**3


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    job_dir = args.job_dir.resolve()
    asset_dir = args.asset_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    started = stamp()
    status = {
        "run_id": RUN_ID,
        "stage": "metadata",
        "status": "STARTING",
        "started_at": started,
        "updated_at": started,
        "ended_at": None,
        "pid": os.getpid(),
        "completed": 0,
        "total": len(ASSETS),
        "bytes_complete": 0,
        "bytes_observed": 0,
        "bytes_expected": None,
        "rate_bytes_per_second": None,
        "eta_seconds": None,
        "exit_code": None,
        "output_path": str(asset_dir),
    }
    atomic_json(job_dir / "status.json", status)
    atomic_json(job_dir / "launch.json", {
        "run_id": RUN_ID,
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "environment": "base Python (network/disk only)",
        "started_at": started,
        "pid": os.getpid(),
        "timeout_seconds": TIMEOUT_SECONDS,
        "disk_expected_max_bytes": 4_000_000_000,
        "disk_free_floor_bytes": FREE_SPACE_FLOOR,
        "source_commit": "eed438bcb49e30405cd6ccdfcccca330c134e830",
        "assets": [
            {"repo": repo, "revision": revision, "file": filename, "directory": folder}
            for repo, revision, filename, folder in ASSETS
        ],
    })
    lock = threading.Lock()
    stopped = threading.Event()
    started_mono = time.monotonic()

    def update(**changes: object) -> None:
        with lock:
            status.update(changes)
            status["updated_at"] = stamp()
            atomic_json(job_dir / "status.json", status)

    def heartbeat() -> None:
        while not stopped.wait(30):
            partial = sum(
                path.stat().st_size
                for path in asset_dir.rglob("*.incomplete")
                if path.is_file()
            )
            observed = min(
                status["bytes_expected"] or 0,
                int(status["bytes_complete"]) + partial,
            )
            elapsed = max(time.monotonic() - started_mono, 1)
            rate = observed / elapsed if observed else None
            remaining = (status["bytes_expected"] or 0) - observed
            update(
                bytes_observed=observed,
                rate_bytes_per_second=rate,
                eta_seconds=remaining / rate if rate and remaining > 0 else None,
            )

    def timeout_handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"download exceeded {TIMEOUT_SECONDS}s")

    def terminate_handler(_signum: int, _frame: object) -> None:
        raise InterruptedError("download was terminated")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGTERM, terminate_handler)
    signal.alarm(TIMEOUT_SECONDS)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    files: list[dict] = []
    try:
        api = HfApi()
        info_cache = {
            (UNISIGN_REPO, UNISIGN_REV): api.model_info(
                UNISIGN_REPO, revision=UNISIGN_REV, files_metadata=True
            ),
            (MT5_REPO, MT5_REV): api.model_info(
                MT5_REPO, revision=MT5_REV, files_metadata=True
            ),
        }
        sizes = {
            (repo, revision, sibling.rfilename): sibling.size
            for (repo, revision), info in info_cache.items()
            for sibling in info.siblings
        }
        expected = sum(sizes[(repo, revision, filename)] for repo, revision, filename, _ in ASSETS)
        update(status="RUNNING", stage="download", bytes_expected=expected)
        print(f"{stamp()} downloading {len(ASSETS)} pinned public files; expected {expected} bytes", flush=True)
        for number, (repo, revision, filename, folder) in enumerate(ASSETS, 1):
            if shutil.disk_usage(asset_dir).free < FREE_SPACE_FLOOR + sizes[(repo, revision, filename)]:
                raise OSError("disk free-space floor would be crossed")
            update(stage=f"download:{filename}")
            print(f"{stamp()} [{number}/{len(ASSETS)}] {repo}@{revision}: {filename}", flush=True)
            path = Path(hf_hub_download(
                repo_id=repo, filename=filename, revision=revision,
                local_dir=asset_dir / folder,
            ))
            actual = path.stat().st_size
            required = sizes[(repo, revision, filename)]
            if actual != required:
                raise ValueError(f"unexpected size for {filename}: {actual} != {required}")
            files.append({
                "repo": repo, "revision": revision, "file": filename,
                "path": str(path), "bytes": actual, "sha256": digest_file(path),
            })
            update(completed=number, bytes_complete=sum(item["bytes"] for item in files))
            print(f"{stamp()} verified {filename}: {actual} bytes, sha256={files[-1]['sha256']}", flush=True)
        summary = {
            "run_id": RUN_ID, "status": "COMPLETED", "exit_code": 0,
            "started_at": started, "ended_at": stamp(), "files": files,
            "licenses": {UNISIGN_REPO: "CC-BY-NC-4.0", MT5_REPO: "Apache-2.0"},
            "gpu_used": False,
        }
        atomic_json(job_dir / "summary.json", summary)
        update(status="COMPLETED", stage="done", ended_at=summary["ended_at"], exit_code=0)
        print(f"{stamp()} completed", flush=True)
        return 0
    except BaseException as exc:
        terminal = (
            "TIMED_OUT" if isinstance(exc, TimeoutError)
            else "CANCELLED" if isinstance(exc, InterruptedError)
            else "FAILED"
        )
        summary = {
            "run_id": RUN_ID, "status": terminal, "exit_code": 1,
            "started_at": started, "ended_at": stamp(), "files": files,
            "error_type": type(exc).__name__, "error": str(exc)[:1000],
            "gpu_used": False,
        }
        atomic_json(job_dir / "summary.json", summary)
        update(status=terminal, ended_at=summary["ended_at"], exit_code=1)
        print(f"{stamp()} {terminal}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.alarm(0)
        stopped.set()
        thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
