from __future__ import annotations

import json

from ocem.cli import NOT_IMPLEMENTED_EXIT, build_parser, main


def test_all_contract_commands_are_registered() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "doctor",
        "resources",
        "data",
        "features",
        "baseline",
        "diagnose",
        "solver",
        "train",
        "evaluate",
        "compare",
        "state",
    ):
        assert command in help_text


def test_future_command_fails_explicitly(capsys) -> None:
    exit_code = main(["train"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == NOT_IMPLEMENTED_EXIT
    assert payload["status"] == "NOT_IMPLEMENTED"
    assert payload["required_work_package"] == "WP-07"


def test_baseline_adapter_validation_command_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "baseline",
            "validate-adapter",
            "--checkpoint",
            "cico.pt",
            "--checkpoint-sha256",
            "a" * 64,
            "--clip-checkpoint",
            "clip.pt",
            "--clip-sha256",
            "b" * 64,
            "--upstream-root",
            "upstream",
            "--modeling-sha256",
            "c" * 64,
            "--module-clip-sha256",
            "d" * 64,
            "--tokenization-sha256",
            "e" * 64,
            "--metrics-sha256",
            "f" * 64,
            "--manifest",
            "train.jsonl",
            "--feature-root",
            "features/train",
            "--sample-id",
            "short",
            "--sample-id",
            "long",
            "--output",
            "parity.json",
        ]
    )
    assert args.handler.__name__ == "_baseline_validate_adapter"


def test_doctor_returns_observed_report(capsys, tmp_path) -> None:
    assert main(["doctor", "--probe-path", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ANALYZED"
    assert payload["probe_path"] == str(tmp_path.resolve())
    assert payload["disk"]["free_bytes"] > 0
