from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import atomic_json_dump, sha256_file, sha256_json


def _command_lines(command: list[str]) -> list[str]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def runtime_environment_report() -> dict[str, Any]:
    requirements = Path(__file__).resolve().parents[1] / "requirements.lock"
    try:
        pip_freeze = _command_lines([sys.executable, "-m", "pip", "freeze"])
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot capture pip freeze: {error}") from error
    try:
        gpu_inventory = _command_lines(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.CalledProcessError):
        gpu_inventory = []
    cudnn = torch.backends.cudnn
    cuda_matmul = torch.backends.cuda.matmul
    target_matches_host = (
        platform.python_version().startswith("3.10.")
        and torch.__version__.split("+")[0] == "2.5.1"
        and np.__version__ == "1.26.4"
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pip_freeze": pip_freeze,
            "pip_freeze_sha256": sha256_json(pip_freeze),
        },
        "cuda": {
            "torch_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "available": torch.cuda.is_available(),
            "gpu_inventory": gpu_inventory,
        },
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": cudnn.deterministic,
            "cudnn_benchmark": cudnn.benchmark,
            "cudnn_allow_tf32": cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": cuda_matmul.allow_tf32,
        },
        "target_lock": {
            "path": str(requirements),
            "sha256": sha256_file(requirements),
            "matches_host_claimed": target_matches_host,
            "note": "Exact target/host differences are explicit; no compatibility equivalence is inferred.",
        },
    }
    report["content_sha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-manifest")
    args = parser.parse_args()
    report = runtime_environment_report()
    atomic_json_dump(report, args.output)
    if args.run_manifest:
        manifest_path = Path(args.run_manifest)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot update run manifest provenance: {error}") from error
        manifest["environment"] = {
            "path": str(Path(args.output).resolve()),
            "sha256": sha256_file(args.output),
        }
        atomic_json_dump(manifest, manifest_path)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
