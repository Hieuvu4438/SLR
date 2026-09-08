from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .adapters import SedsAdapterError, SedsDataError, SedsReproductionError
from .artifacts import ArtifactError
from .baseline import BaselineValidationError, validate_seds_baseline
from .baseline_training import BaselineTrainingError, train_seds_baseline
from .cache import CacheError
from .cache_build import CacheBuildError, build_frozen_train_cache
from .config import ConfigError, dump_resolved, load_config
from .data.manifest import ManifestError
from .data.prepare import PreparationError, prepare_how2sign_data
from .data.relations import RelationError
from .data.relevance import RelevanceError
from .data.temporal import TemporalError
from .data.text_units import TextUnitError
from .data.validation import DataValidationError, validate_prepared_data
from .doctor import inspect_environment, write_report
from .evidence_warmup import EvidenceWarmupError, warmup_seds_evidence
from .mining.neighbors import NeighborError
from .mining.runner import MiningRunError, propose_train_contrasts
from .models.evidence import EvidenceError
from .smoke import run_fixture_smoke
from .training.optimizer import OptimizerContractError
from .training.state import CheckpointError
from .training.warmup import WarmupContractError


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
            "fixture",
            "prepare",
            "baseline_train",
            "baseline_validate",
            "warmup",
            "cache_build",
            "validate_data",
            "mine_propose",
            "mine_finalize",
            "evaluate_test",
        ),
    )
    doctor.add_argument("--output", default="doctor.json")

    config = subparsers.add_parser("config", help="validate and resolve a YAML config")
    config.add_argument("--config", required=True)
    config.add_argument("--output")

    validate_data = subparsers.add_parser(
        "validate-data", help="validate prepared manifests, relevance, relations, and assets"
    )
    validate_data.add_argument("--config", required=True)
    validate_data.add_argument("--output")

    prepare_data = subparsers.add_parser(
        "prepare-data", help="prepare controlled How2Sign manifests and temporal maps"
    )
    prepare_data.add_argument("--config", required=True)
    prepare_data.add_argument("--workers", type=int)

    baseline = subparsers.add_parser("baseline", help="train or validate the controlled B0")
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_validate = baseline_commands.add_parser(
        "validate", help="validate locked SEDS B0 on the controlled dev gallery"
    )
    baseline_validate.add_argument("--config", required=True)
    baseline_validate.add_argument("--split", choices=("dev",), default="dev")
    baseline_validate.add_argument("--batch-size", type=int, default=64)
    baseline_validate.add_argument("--device", default="cuda:0")
    baseline_train = baseline_commands.add_parser(
        "train", help="train and dev-select the controlled native SEDS B0"
    )
    baseline_train.add_argument("--config", required=True)
    baseline_train.add_argument("--batch-size", type=int, default=128)
    baseline_train.add_argument("--eval-batch-size", type=int, default=64)
    baseline_train.add_argument("--device", default="cuda:0")
    baseline_train.add_argument("--resume", action="store_true")

    evidence = subparsers.add_parser("evidence", help="train the DIVE local evidence model")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    evidence_warmup = evidence_commands.add_parser(
        "warmup", help="warm up and dev-select the frozen local reference"
    )
    evidence_warmup.add_argument("--config", required=True)
    evidence_warmup.add_argument("--batch-size", type=int, default=128)
    evidence_warmup.add_argument("--eval-video-batch-size", type=int)
    evidence_warmup.add_argument("--eval-text-batch-size", type=int)
    evidence_warmup.add_argument("--device", default="cuda:0")
    evidence_warmup.add_argument("--resume", action="store_true")

    cache = subparsers.add_parser("cache", help="build checksummed frozen tensor caches")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_build = cache_commands.add_parser("build", help="build a frozen feature cache")
    cache_build.add_argument("--config", required=True)
    cache_build.add_argument("--kind", required=True, choices=("frozen_train",))
    cache_build.add_argument("--batch-size", type=int, default=32)
    cache_build.add_argument("--device", default="cuda:0")

    mine = subparsers.add_parser("mine", help="mine and validate train-only contrast pairs")
    mine.add_argument("--config", required=True)
    mine.add_argument("--phase", required=True, choices=("propose",))
    mine.add_argument("--device", default="cuda:0")
    mine.add_argument("--pooled-query-chunk-size", type=int, default=512)
    mine.add_argument("--pair-chunk-size", type=int, default=2048)

    smoke = subparsers.add_parser("smoke", help="run fixture-only end-to-end correctness smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--output-dir", required=True)
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
        if args.command == "validate-data":
            report = validate_prepared_data(config, output_path=args.output)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "prepare-data":
            report = prepare_how2sign_data(config, workers=args.workers)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "baseline" and args.baseline_command == "validate":
            report = validate_seds_baseline(
                config,
                split=args.split,
                batch_size=args.batch_size,
                device=args.device,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "baseline" and args.baseline_command == "train":
            report = train_seds_baseline(
                config,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                device=args.device,
                resume=args.resume,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "evidence" and args.evidence_command == "warmup":
            report = warmup_seds_evidence(
                config,
                batch_size=args.batch_size,
                eval_video_batch_size=args.eval_video_batch_size,
                eval_text_batch_size=args.eval_text_batch_size,
                device=args.device,
                resume=args.resume,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "cache" and args.cache_command == "build":
            report = build_frozen_train_cache(
                config,
                batch_size=args.batch_size,
                device=args.device,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "mine" and args.phase == "propose":
            report = propose_train_contrasts(
                config,
                device=args.device,
                pooled_query_chunk_size=args.pooled_query_chunk_size,
                pair_chunk_size=args.pair_chunk_size,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "smoke":
            root = Path(__file__).resolve().parents[4]
            report = run_fixture_smoke(config, args.output_dir, repository_root=root)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
    except ConfigError as exc:
        print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
        return 2
    except (
        ArtifactError,
        BaselineTrainingError,
        BaselineValidationError,
        CacheBuildError,
        CacheError,
        CheckpointError,
        DataValidationError,
        EvidenceWarmupError,
        EvidenceError,
        ManifestError,
        MiningRunError,
        NeighborError,
        PreparationError,
        RelevanceError,
        RelationError,
        SedsAdapterError,
        SedsDataError,
        SedsReproductionError,
        TemporalError,
        TextUnitError,
        OptimizerContractError,
        WarmupContractError,
    ) as exc:
        print(f"DATA_ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
