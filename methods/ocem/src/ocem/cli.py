"""Command-line interface for the staged OCEM implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ocem.config import ConfigError, load_config
from ocem.data.adaptation import AdaptationPlanError, build_p14t_adaptation_plan
from ocem.data.datasets.how2sign import prepare_how2sign
from ocem.data.datasets.phoenix import prepare_phoenix
from ocem.data.extraction import ExtractionRunError, extract_p14t_i3d_features
from ocem.data.features import FeatureAuditError, audit_feature_cache
from ocem.data.pseudoclips import PseudoClipError, validate_pseudoclip_loader
from ocem.data.pseudolabels import PseudoLabelError, generate_p14t_pseudolabel_index
from ocem.doctor import collect_doctor_report
from ocem.provenance.resources import build_resource_lock
from ocem.provenance.state import load_implementation_state, render_checkpoint
from ocem.scoring.validation import profile_solver, validate_solver
from ocem.training.adaptation_run import AdaptationRunError, run_p14t_i3d_adaptation
from ocem.training.i3d_adaptation import (
    I3DAdaptationError,
    validate_one_batch_adaptation_parity,
)


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


def _solver_validate(args: argparse.Namespace) -> int:
    if args.reference:
        expected = Path(args.reference).resolve()
        bundled = Path(__file__).with_name("scoring") / "reference.py"
        if expected != bundled.resolve():
            raise ConfigError(
                f"--reference must resolve to the immutable bundled oracle {bundled.resolve()}"
            )
    report = validate_solver(device=args.device, dtype=args.dtype)
    if args.profile:
        report["profile"] = profile_solver(
            device=args.device, dtype=args.dtype, pairs=args.profile_pairs
        )
        if report["profile"]["status"] != "PASS":
            report["status"] = "FAIL_TECHNICAL"
    _write_json(report, args.output)
    return 0 if report["status"] == "PASS" else 5


def _features_audit_cache(args: argparse.Namespace) -> int:
    if args.dataset != "phoenix2014t":
        raise FeatureAuditError("only phoenix2014t cache auditing is implemented")
    report = audit_feature_cache(
        manifest_dir=args.manifest_dir,
        feature_root=args.feature_root,
        temporal_root=args.temporal_root,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        split_dirs={"train": "train", "validation": "dev", "test": "test"},
        workers=args.workers,
        expected_stream_name=args.stream_name,
        adaptation_report=args.adaptation_report,
        expected_adaptation_report_sha256=args.adaptation_report_sha256,
    )
    _write_json(report, args.output)
    return 0 if report["status"] == "PASS" else 4


def _features_plan_adaptation(args: argparse.Namespace) -> int:
    report = build_p14t_adaptation_plan(
        train_manifest=args.train_manifest,
        forbidden_manifests=args.forbidden_manifest,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        class_vocabulary=args.class_vocabulary,
        seed=args.seed,
        holdout_modulus=args.holdout_modulus,
    )
    _write_json(report, args.output)
    return 0


def _features_pseudolabel(args: argparse.Namespace) -> int:
    report = generate_p14t_pseudolabel_index(
        adaptation_plan=args.adaptation_plan,
        expected_plan_sha256=args.plan_sha256,
        feature_dir=args.feature_dir,
        temporal_dir=args.temporal_dir,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        output_index=args.output_index,
        device=args.device,
        batch_windows=args.batch_windows,
        report_interval=args.report_interval,
    )
    _write_json(report, args.output)
    return 0 if report["status"] == "PASS" else 4


def _features_validate_pseudoclips(args: argparse.Namespace) -> int:
    report = validate_pseudoclip_loader(
        index=args.index,
        expected_index_sha256=args.index_sha256,
        adaptation_split=args.adaptation_split,
        samples=args.samples,
        seed=args.seed,
    )
    _write_json(report, args.output)
    return 0 if report["status"] == "PASS" else 4


def _features_validate_adaptation_step(args: argparse.Namespace) -> int:
    report = validate_one_batch_adaptation_parity(
        index=args.index,
        expected_index_sha256=args.index_sha256,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        implementation=args.implementation,
        expected_implementation_sha256=args.implementation_sha256,
        trainer_root=args.trainer_root,
        expected_transforms_sha256=args.transforms_sha256,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        absolute_tolerance=args.atol,
        relative_tolerance=args.rtol,
    )
    _write_json(report, args.output)
    return 0 if report["status"] == "PASS" else 5


def _features_adapt(args: argparse.Namespace) -> int:
    report = run_p14t_i3d_adaptation(
        index=args.index,
        expected_index_sha256=args.index_sha256,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        implementation=args.implementation,
        expected_implementation_sha256=args.implementation_sha256,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        snapshot_interval=args.snapshot_interval,
        log_interval=args.log_interval,
        resume=args.resume,
    )
    _write_json(report, args.output)
    return 0 if report["status"] == "PASS" else 5


def _features_extract(args: argparse.Namespace) -> int:
    report = extract_p14t_i3d_features(
        manifest_dir=args.manifest_dir,
        video_root=args.video_root,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        temporal_root=args.temporal_root,
        shared_root=args.shared_root,
        extractor=args.extractor,
        expected_extractor_sha256=args.extractor_sha256,
        upstream_i3d=args.upstream_i3d,
        expected_upstream_i3d_sha256=args.upstream_i3d_sha256,
        splits=args.split,
        device=args.device,
        batch_size=args.batch_size,
        min_free_disk_gib=args.min_free_disk_gib,
        min_free_gpu_gib=args.min_free_gpu_gib,
        report_interval=args.report_interval,
        dry_run=args.dry_run,
    )
    _write_json(report, args.output)
    return 0 if report["status"] in {"PASS", "PLANNED"} else 5


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
    validate = config_commands.add_parser(
        "validate", help="Strictly parse and validate a YAML file."
    )
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
    features = commands.add_parser(
        "features", help="Adapt, extract, and audit frozen I3D features."
    )
    feature_commands = features.add_subparsers(dest="features_command", required=True)
    extract = feature_commands.add_parser(
        "extract", help="Extract manifest-scoped P14T adapted I3D features."
    )
    extract.add_argument("--manifest-dir", required=True)
    extract.add_argument("--video-root", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--checkpoint-sha256", required=True)
    extract.add_argument("--output-root", required=True)
    extract.add_argument("--temporal-root", required=True)
    extract.add_argument("--shared-root", required=True)
    extract.add_argument("--extractor", required=True)
    extract.add_argument("--extractor-sha256", required=True)
    extract.add_argument("--upstream-i3d", required=True)
    extract.add_argument("--upstream-i3d-sha256", required=True)
    extract.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        required=True,
    )
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--min-free-disk-gib", type=float, default=20.0)
    extract.add_argument("--min-free-gpu-gib", type=float, default=12.0)
    extract.add_argument("--report-interval", type=int, default=25)
    extract.add_argument("--dry-run", action="store_true")
    extract.add_argument("--output")
    extract.set_defaults(handler=_features_extract)
    adapt = feature_commands.add_parser("adapt", help="Run locked P14T train-only adaptation.")
    adapt.add_argument("--index", required=True)
    adapt.add_argument("--index-sha256", required=True)
    adapt.add_argument("--checkpoint", required=True)
    adapt.add_argument("--checkpoint-sha256", required=True)
    adapt.add_argument("--implementation", required=True)
    adapt.add_argument("--implementation-sha256", required=True)
    adapt.add_argument("--output-dir", required=True)
    adapt.add_argument("--device", default="cuda:0")
    adapt.add_argument("--epochs", type=int, default=15)
    adapt.add_argument("--batch-size", type=int, default=4)
    adapt.add_argument("--workers", type=int, default=4)
    adapt.add_argument("--seed", type=int, default=0)
    adapt.add_argument("--snapshot-interval", type=int, default=5)
    adapt.add_argument("--log-interval", type=int, default=100)
    adapt.add_argument("--resume", action="store_true")
    adapt.add_argument("--output")
    adapt.set_defaults(handler=_features_adapt)
    audit_cache = feature_commands.add_parser(
        "audit-cache", help="Audit an existing I3D cache against protocol manifests."
    )
    audit_cache.add_argument("--dataset", required=True, choices=("phoenix2014t",))
    audit_cache.add_argument("--manifest-dir", required=True)
    audit_cache.add_argument("--feature-root", required=True)
    audit_cache.add_argument("--temporal-root", required=True)
    audit_cache.add_argument("--checkpoint-sha256", required=True)
    audit_cache.add_argument(
        "--stream-name",
        default="domain_agnostic",
        choices=("domain_agnostic", "domain_adapted_p14t"),
    )
    audit_cache.add_argument("--adaptation-report")
    audit_cache.add_argument("--adaptation-report-sha256")
    audit_cache.add_argument("--workers", type=int, default=8)
    audit_cache.add_argument("--output", required=True)
    audit_cache.set_defaults(handler=_features_audit_cache)
    plan_adaptation = feature_commands.add_parser(
        "plan-adaptation", help="Lock P14T train-only adaptation IDs and recipe."
    )
    plan_adaptation.add_argument("--train-manifest", required=True)
    plan_adaptation.add_argument("--forbidden-manifest", action="append", required=True)
    plan_adaptation.add_argument("--checkpoint", required=True)
    plan_adaptation.add_argument("--checkpoint-sha256", required=True)
    plan_adaptation.add_argument("--class-vocabulary", required=True)
    plan_adaptation.add_argument("--seed", type=int, default=0)
    plan_adaptation.add_argument("--holdout-modulus", type=int, default=10)
    plan_adaptation.add_argument("--output", required=True)
    plan_adaptation.set_defaults(handler=_features_plan_adaptation)
    pseudolabel = feature_commands.add_parser(
        "pseudolabel", help="Generate a train-only P14T pseudo-label segment index."
    )
    pseudolabel.add_argument("--adaptation-plan", required=True)
    pseudolabel.add_argument("--plan-sha256", required=True)
    pseudolabel.add_argument("--feature-dir", required=True)
    pseudolabel.add_argument("--temporal-dir", required=True)
    pseudolabel.add_argument("--checkpoint", required=True)
    pseudolabel.add_argument("--checkpoint-sha256", required=True)
    pseudolabel.add_argument("--output-index", required=True)
    pseudolabel.add_argument("--device", default="cuda:0")
    pseudolabel.add_argument("--batch-windows", type=int, default=8192)
    pseudolabel.add_argument("--report-interval", type=int, default=250)
    pseudolabel.add_argument("--output", required=True)
    pseudolabel.set_defaults(handler=_features_pseudolabel)
    validate_pseudoclips = feature_commands.add_parser(
        "validate-pseudoclips", help="Smoke-test raw-frame pseudo clips from a locked index."
    )
    validate_pseudoclips.add_argument("--index", required=True)
    validate_pseudoclips.add_argument("--index-sha256", required=True)
    validate_pseudoclips.add_argument(
        "--adaptation-split", required=True, choices=("train", "holdout")
    )
    validate_pseudoclips.add_argument("--samples", type=int, default=4)
    validate_pseudoclips.add_argument("--seed", type=int, default=0)
    validate_pseudoclips.add_argument("--output", required=True)
    validate_pseudoclips.set_defaults(handler=_features_validate_pseudoclips)
    validate_adaptation_step = feature_commands.add_parser(
        "validate-adaptation-step",
        help="Compare one P14T I3D SGD update with the pinned CiCo path.",
    )
    validate_adaptation_step.add_argument("--index", required=True)
    validate_adaptation_step.add_argument("--index-sha256", required=True)
    validate_adaptation_step.add_argument("--checkpoint", required=True)
    validate_adaptation_step.add_argument("--checkpoint-sha256", required=True)
    validate_adaptation_step.add_argument("--implementation", required=True)
    validate_adaptation_step.add_argument("--implementation-sha256", required=True)
    validate_adaptation_step.add_argument("--trainer-root", required=True)
    validate_adaptation_step.add_argument("--transforms-sha256", required=True)
    validate_adaptation_step.add_argument("--device", default="cuda:0")
    validate_adaptation_step.add_argument("--batch-size", type=int, default=4)
    validate_adaptation_step.add_argument("--seed", type=int, default=0)
    validate_adaptation_step.add_argument("--atol", type=float, default=1e-5)
    validate_adaptation_step.add_argument("--rtol", type=float, default=1e-4)
    validate_adaptation_step.add_argument("--output", required=True)
    validate_adaptation_step.set_defaults(handler=_features_validate_adaptation_step)
    baseline = commands.add_parser("baseline", help="Operate the pinned CiCo baseline.")
    _nested_placeholder(
        baseline, "baseline", [("reproduce", "Run baseline reproduction.", "WP-08")]
    )
    diagnose = commands.add_parser("diagnose", help="Run preregistered mechanism diagnostics.")
    _nested_placeholder(
        diagnose,
        "diagnose",
        [("concentration", "Measure support concentration for Stage A/A2.", "WP-09")],
    )
    solver = commands.add_parser("solver", help="Validate the certified OCEM solver.")
    solver_commands = solver.add_subparsers(dest="solver_command", required=True)
    solver_validate = solver_commands.add_parser(
        "validate", help="Run solver parity and profiling."
    )
    solver_validate.add_argument("--reference")
    solver_validate.add_argument("--device", default="cpu")
    solver_validate.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    solver_validate.add_argument("--profile", action="store_true")
    solver_validate.add_argument("--profile-pairs", type=int, default=2)
    solver_validate.add_argument("--output", required=True)
    solver_validate.set_defaults(handler=_solver_validate)

    _placeholder(commands, "train", "Train an eligible matched experiment.", "WP-07")
    _placeholder(commands, "evaluate", "Run full-gallery evaluation.", "WP-07")
    _placeholder(commands, "compare", "Compare equivalent runs and bootstrap differences.", "WP-07")

    state = commands.add_parser("state", help="Inspect or checkpoint implementation state.")
    state_commands = state.add_subparsers(dest="state_command", required=True)
    checkpoint = state_commands.add_parser(
        "checkpoint", help="Render a resumable Markdown checkpoint."
    )
    checkpoint.add_argument("--state", default="implementation_state.json")
    checkpoint.add_argument("--output", required=True)
    checkpoint.set_defaults(handler=_state_checkpoint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ConfigError,
        FeatureAuditError,
        ExtractionRunError,
        AdaptationPlanError,
        AdaptationRunError,
        I3DAdaptationError,
        PseudoClipError,
        PseudoLabelError,
    ) as error:
        sys.stderr.write(f"configuration error: {error}\n")
        return 2
