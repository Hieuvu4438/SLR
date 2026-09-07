from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def atomic_json_dump(value: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def ordered_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def git_worktree_state(path: str | Path) -> dict[str, Any]:
    """Capture the checked-out implementation revision without mutating Git state."""
    probe = Path(path).resolve()
    cwd = probe if probe.is_dir() else probe.parent
    try:
        root = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return {"status": "unavailable", "error": type(error).__name__}
    return {
        "status": "ready",
        "root": root,
        "commit": commit,
        "tracked_worktree_dirty": bool(status.strip()),
    }
