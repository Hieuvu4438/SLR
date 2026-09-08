from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import ConfigError, dump_resolved, load_config
from .doctor import inspect_environment, write_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dive", description="DIVE-SLR v2 research pipeline")
    parser.add_argument("--version", action="version", version="dive 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect environment and stage resources")
    doctor.add_argument("--config", required=True)
    doctor.add_argument(
        "--stage",
        required=True,
        choices=(
            "fixture", "prepare", "baseline_train", "baseline_validate", "warmup",
            "mine_finalize", "evaluate_test",
        ),
    )
    doctor.add_argument("--output", default="doctor.json")

    config = subparsers.add_parser("config", help="validate and resolve a YAML config")
    config.add_argument("--config", required=True)
    config.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            report = inspect_environment(config, args.stage)
            write_report(report, args.output)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ready"] else 2
        if args.command == "config":
            digest = dump_resolved(config, args.output) if args.output else None
            print(digest or json.dumps(config, sort_keys=True))
            return 0
    except ConfigError as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
