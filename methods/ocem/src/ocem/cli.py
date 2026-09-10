"""Command-line interface for the staged OCEM implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ocem.config import ConfigError, load_config
from ocem.data.datasets.how2sign import prepare_how2sign
from ocem.data.datasets.phoenix import prepare_phoenix
from ocem.doctor import collect_doctor_report
from ocem.provenance.resources import build_resource_lock
from ocem.provenance.state import load_implementation_state, render_checkpoint


NOT_IMPLEMENTED_EXIT = 3


def _write_json(payload: Any, output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _not_implemented(args: argparse.Namespace) -> int:
    command = " ".join(args.command_path)
    _write_json(
        {
            "schema_version": "ocem.command_status.v1",
            "command": command,
            "status": "NOT_IMPLEMENTED",
            "required_work_package": args.required_work_package,
        },
        getattr(args, "output", None),
    )
    return NOT_IMPLEMENTED_EXIT


def _doctor(args: argparse.Namespace) -> int:
    if args.config:
        load_config(args.config, kind="resources")
    _write_json(collect_doctor_report(args.probe_path), args.output)
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = load_config(args.path, kind=args.kind)
    _write_json(
        {
            "schema_version": "ocem.config_validation.v1",
            "status": "PASS",
            "kind": args.kind,
            "path": str(Path(args.path).resolve()),
            "resolved_schema_version": config["schema_version"],
        },
        args.output,
    )
    return 0


def _resources_verify(args: argparse.Namespace) -> int:
    config = load_config(args.config, kind="resources")
    lock = build_resource_lock(config, args.config)
    _write_json(lock, args.output)
    return 0 if lock["status"] == "PASS" else 4


def _data_prepare(args: argparse.Namespace) -> int:
    protocol = load_config(args.protocol, kind="protocol")
    if protocol["dataset"] != args.dataset:
        raise ConfigError(
            f"protocol dataset {protocol['dataset']!r} does not match --dataset {args.dataset!r}"
        )
    if args.dataset == "phoenix2014t":
        report = prepare_phoenix(protocol, args.output_dir, workers=args.workers)
    else:
        report = prepare_how2sign(protocol, args.output_dir, workers=args.workers)
    _write_json(report, args.report)
    return 0 if report["status"] == "PASS" else 4


def _state_checkpoint(args: argparse.Namespace) -> int:
    state = load_implementation_state(args.state)
    checkpoint = render_checkpoint(state)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(checkpoint, encoding="utf-8")
    return 0


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="Write JSON output to this path instead of stdout.")


def _placeholder(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    work_package: str,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=help_text)
    _add_output(parser)
    parser.set_defaults(
        handler=_not_implemented,
        command_path=[name],
        required_work_package=work_package,
    )
    return parser


def _nested_placeholder(
    parent: argparse.ArgumentParser,
    parent_name: str,
    children: Sequence[tuple[str, str, str]],
) -> None:
    subparsers = parent.add_subparsers(dest=f"{parent_name}_command", required=True)
    for name, help_text, work_package in children:
        parser = subparsers.add_parser(name, help=help_text)
        _add_output(parser)
        parser.set_defaults(
            handler=_not_implemented,
            command_path=[parent_name, name],
            required_work_package=work_package,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocem",
        description="Gate-aware OCEM research implementation. No command bypasses prerequisites.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Inspect environment, GPU, storage, and packages.")
    doctor.add_argument("--config", help="Optional strict resource registry YAML.")
    doctor.add_argument("--probe-path", default=".", help="Path whose filesystem is inspected.")
    _add_output(doctor)
    doctor.set_defaults(handler=_doctor)

    config = commands.add_parser("config", help="Validate configuration contracts.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="Strictly parse and validate a YAML file.")
    validate.add_argument("--kind", required=True, choices=("resources", "protocol", "experiment"))
    validate.add_argument("path")
    _add_output(validate)
    validate.set_defaults(handler=_validate_config)

    resources = commands.add_parser("resources", help="Verify explicit local resource artifacts.")
    resource_commands = resources.add_subparsers(dest="resources_command", required=True)
    verify = resource_commands.add_parser("verify", help="Verify hashes/content/load state.")
    verify.add_argument("--config", required=True)
    verify.add_argument("--output", required=True)
    verify.set_defaults(handler=_resources_verify)
    data = commands.add_parser("data", help="Build provenance-preserving dataset manifests.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    prepare = data_commands.add_parser("prepare", help="Prepare a dataset manifest.")
    prepare.add_argument("--dataset", required=True, choices=("phoenix2014t", "how2sign"))
    prepare.add_argument("--protocol", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--report")
    prepare.set_defaults(handler=_data_prepare)
    features = commands.add_parser("features", help="Adapt and extract frozen I3D features.")
    _nested_placeholder(
        features,
        "features",
        [
            ("adapt", "Run train-only target-domain adaptation.", "WP-04"),
            ("extract", "Extract feature shards and temporal supports.", "WP-04"),
        ],
    )
    baseline = commands.add_parser("baseline", help="Operate the pinned CiCo baseline.")
    _nested_placeholder(baseline, "baseline", [("reproduce", "Run baseline reproduction.", "WP-08")])
    diagnose = commands.add_parser("diagnose", help="Run preregistered mechanism diagnostics.")
    _nested_placeholder(
        diagnose,
        "diagnose",
        [("concentration", "Measure support concentration for Stage A/A2.", "WP-09")],
    )
    solver = commands.add_parser("solver", help="Validate the certified OCEM solver.")
    _nested_placeholder(solver, "solver", [("validate", "Run solver parity and profiling.", "WP-06")])
    _placeholder(commands, "train", "Train an eligible matched experiment.", "WP-07")
    _placeholder(commands, "evaluate", "Run full-gallery evaluation.", "WP-07")
    _placeholder(commands, "compare", "Compare equivalent runs and bootstrap differences.", "WP-07")

    state = commands.add_parser("state", help="Inspect or checkpoint implementation state.")
    state_commands = state.add_subparsers(dest="state_command", required=True)
    checkpoint = state_commands.add_parser("checkpoint", help="Render a resumable Markdown checkpoint.")
    checkpoint.add_argument("--state", default="implementation_state.json")
    checkpoint.add_argument("--output", required=True)
    checkpoint.set_defaults(handler=_state_checkpoint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ConfigError as error:
        sys.stderr.write(f"configuration error: {error}\n")
        return 2
