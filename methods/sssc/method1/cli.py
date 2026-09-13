from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import numpy as np
import torch

from .audit import audit_resources
from .config import ConfigError, load_config
from .comparison import compare_runs
from .diagnostics import support_diagnostics
from .evaluation import evaluate_grouped_retrieval
from .inference import evaluate_checkpoint, export_student
from .losses.shared_support import span_contrast_terms
from .manifests import build_manifests
from .mining_pipeline import mine_reference_negatives
from .reference_pipeline import create_reference_cache
from .sampling import mix_and_sample_features
from .token_spans import tokenize_with_spans
from .train_pipeline import train_stage
from .upstream import create_upret_tokenizer
from .utils import atomic_json_dump


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m method1.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="validate command-scoped resources")
    audit.add_argument("--config", required=True)
    audit.add_argument("--stage", choices=("input", "method", "inference"), required=True)
    audit.add_argument("--output")

    manifests = subparsers.add_parser("build-manifests", help="build canonical JSONL manifests")
    manifests.add_argument("--config", required=True)

    tokenizer = subparsers.add_parser("verify-tokenizer", help="verify pinned tokenizer contracts")
    tokenizer.add_argument("--config", required=True)
    tokenizer.add_argument("--upret-root", default="third_party/UPRet")
    tokenizer.add_argument("--output")

    smoke = subparsers.add_parser("smoke", help="run bounded synthetic tensor checks")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--synthetic", action="store_true", required=True)
    smoke.add_argument("--output")

    # Register every normative entry point now. Commands not yet safe to execute fail closed
    # with a machine-readable status instead of silently substituting another workflow.
    for name in (
        "train-base",
        "cache-reference",
        "mine-negatives",
        "diagnose",
        "train-method",
        "evaluate",
        "export",
        "compare-runs",
    ):
        command = subparsers.add_parser(name)
        if name == "compare-runs":
            command.add_argument("--runs", nargs=2, required=True)
        else:
            command.add_argument("--config", required=True)
        if name in {"train-base", "train-method"}:
            command.add_argument("--max-steps", type=int)
            command.add_argument("--resume")
            command.add_argument("--device", default="auto")
            command.add_argument("--upret-root", default="third_party/UPRet")
        if name in {"cache-reference", "mine-negatives"}:
            command.add_argument("--device", default="auto")
            command.add_argument("--upret-root", default="third_party/UPRet")
            command.add_argument("--batch-size", type=int)
        if name in {"evaluate", "export"}:
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--device", default="auto")
            command.add_argument("--upret-root", default="third_party/UPRet")
        if name in {"evaluate", "diagnose"}:
            command.add_argument("--split", choices=("dev", "test"), required=True)
        if name == "diagnose":
            command.add_argument("--kind", choices=("support",), required=True)
            command.add_argument("--device", default="auto")
            command.add_argument("--upret-root", default="third_party/UPRet")
        if name == "export":
            command.add_argument("--output", required=True)
    return parser


def _emit(report: dict[str, Any], output: str | None = None) -> None:
    if output:
        atomic_json_dump(report, output)
    serializable = dict(report)
    # Synthetic grouped scores are returned for tests but should not be printed as ndarray.
    if isinstance(serializable.get("grouped_t2v_scores"), np.ndarray):
        serializable["grouped_t2v_scores"] = serializable["grouped_t2v_scores"].tolist()
    print(json.dumps(serializable, ensure_ascii=False, sort_keys=True))


def _verify_tokenizer(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    tokenizer = create_upret_tokenizer(args.upret_root, config.model.bpe_path)
    fixtures = [
        "Rain, then rain.",
        "We're going to work on graceful hand movements in front of you.",
        "Café&nbsp; weather — tomorrow",
        " ".join(f"word{index}" for index in range(80)),
    ]
    summaries = []
    for index, text in enumerate(fixtures):
        encoded = tokenize_with_spans(
            text,
            text_uid=f"fixture:{index}",
            tokenizer=tokenizer,
            max_positions=config.data.text_max_positions,
        )
        summaries.append(
            {
                "fixture": index,
                "full_bpe_count": len(encoded.full_bpe_ids),
                "fully_retained": encoded.fully_retained,
                "lexical_spans": len(encoded.lexical_spans),
            }
        )
    return {"schema_version": 1, "status": "pass", "fixtures": summaries}


def _synthetic_smoke(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    agnostic = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    aware = np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32)
    mixed, valid, _ = mix_and_sample_features(
        agnostic, aware, agnostic_weight=config.data.agnostic_weight, feature_len=2
    )
    expected = config.data.agnostic_weight * agnostic + (1 - config.data.agnostic_weight) * aware
    if not np.allclose(mixed, expected) or not valid.all():
        raise AssertionError("synthetic feature mixture failed")

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    q_pos = torch.tensor([[[[1.0, 0.0]]]])
    q_neg = torch.tensor([[[[0.0, 1.0]]]])
    tensors = (
        x,
        x.detach().clone(),
        q_pos,
        q_neg,
        torch.tensor([[True, True]]),
        torch.tensor([[[True]]]),
        torch.ones(1, 1, 1),
    )
    shared = span_contrast_terms(*tensors, mode="shared", tau=0.07, margin=1.1)
    independent = span_contrast_terms(*tensors, mode="independent", tau=0.07, margin=1.1)
    shared.numerator.backward()
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise AssertionError("synthetic auxiliary gradient failed")
    retrieval = evaluate_grouped_retrieval(
        np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.95]]),
        video_group_indexes=[0, 0, 1],
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "label": "smoke",
        "config_sha256": config.digest,
        "checks": {
            "feature_mixture": "pass",
            "shared_margin": float(shared.diagnostics["delta"].item()),
            "independent_margin": float(independent.diagnostics["delta"].item()),
            "video_gradient_norm": float(x.grad.norm().item()),
            "grouped_retrieval": {
                "V2T_ranks": retrieval["V2T"]["ranks"],
                "T2V_ranks": retrieval["T2V"]["ranks"],
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit":
            report = audit_resources(load_config(args.config), args.stage)
            _emit(report, args.output)
        elif args.command == "build-manifests":
            report = build_manifests(load_config(args.config))
            _emit(report)
        elif args.command == "verify-tokenizer":
            _emit(_verify_tokenizer(args), args.output)
        elif args.command == "smoke":
            _emit(_synthetic_smoke(args.config), args.output)
        elif args.command == "cache-reference":
            config = load_config(args.config)
            device = (
                "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
            )
            if device == "auto":
                device = "cpu"
            _emit(
                create_reference_cache(
                    config,
                    upret_root=args.upret_root,
                    device=device,
                    batch_size=args.batch_size,
                )
            )
        elif args.command == "mine-negatives":
            config = load_config(args.config)
            device = (
                "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
            )
            if device == "auto":
                device = "cpu"
            _emit(
                mine_reference_negatives(
                    config,
                    upret_root=args.upret_root,
                    device=device,
                    batch_size=args.batch_size,
                )
            )
        elif args.command in {"train-base", "train-method"}:
            _emit(
                train_stage(
                    load_config(args.config),
                    stage="base" if args.command == "train-base" else "method",
                    max_steps=args.max_steps,
                    resume=args.resume,
                    requested_device=args.device,
                    upret_root=args.upret_root,
                )
            )
        elif args.command == "evaluate":
            config = load_config(args.config)
            device = (
                "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
            )
            if device == "auto":
                device = "cpu"
            _emit(
                evaluate_checkpoint(
                    config,
                    args.checkpoint,
                    split=args.split,
                    upret_root=args.upret_root,
                    device=device,
                )
            )
        elif args.command == "export":
            _emit(
                export_student(
                    load_config(args.config),
                    args.checkpoint,
                    args.output,
                    upret_root=args.upret_root,
                )
            )
        elif args.command == "diagnose":
            config = load_config(args.config)
            device = (
                "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
            )
            if device == "auto":
                device = "cpu"
            _emit(
                support_diagnostics(
                    config,
                    split=args.split,
                    upret_root=args.upret_root,
                    device=device,
                )
            )
        elif args.command == "compare-runs":
            _emit(compare_runs(args.runs[0], args.runs[1]))
        else:
            _emit(
                {
                    "schema_version": 1,
                    "status": "unimplemented",
                    "command": args.command,
                    "reason": "command is registered but its correctness milestone has not passed",
                }
            )
            return 3
    except (ConfigError, ValueError, OSError, RuntimeError, AssertionError) as error:
        _emit(
            {
                "schema_version": 1,
                "status": "failed",
                "command": args.command,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
