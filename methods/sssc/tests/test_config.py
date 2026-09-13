from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from method1.config import ConfigError, load_config


CONFIG = Path("methods/sssc/configs/method1/ph_span_shared.yaml")
BASE_CONFIG = Path("methods/sssc/configs/method1/ph_base_initial.yaml")


def test_config_is_strict_and_resolves_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLRET_DATA_ROOT", "/tmp/method1-data")
    monkeypatch.setenv("SLRET_RUN_ROOT", "/tmp/method1-runs")
    config = load_config(CONFIG)
    assert config.data.dataset == "ph"
    assert config.data.agnostic_weight == 0.9
    assert config.auxiliary.arm == "span_shared"
    assert config.model.clip_checkpoint_path == "/tmp/method1-data/initialization/ViT-B-32.pt"
    assert len(config.digest) == 64


def test_config_rejects_unresolved_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLRET_DATA_ROOT", raising=False)
    monkeypatch.setenv("SLRET_RUN_ROOT", "/tmp/method1-runs")
    with pytest.raises(ConfigError, match="SLRET_DATA_ROOT"):
        load_config(CONFIG)


def test_local_base_initial_config_is_explicit_and_valid() -> None:
    config = load_config(BASE_CONFIG)
    assert config.auxiliary.arm == "base_initial"
    assert config.output.root.endswith("/base/seed42")


def test_k1_pilot_pair_differs_only_in_declared_intervention_and_output() -> None:
    independent = load_config(
        "methods/sssc/configs/method1/ph_seed42_span_independent_k1_pilot.yaml"
    )
    shared = load_config(
        "methods/sssc/configs/method1/ph_seed42_span_shared_k1_pilot.yaml"
    )
    assert independent.auxiliary.negatives_per_caption == 1
    assert shared.auxiliary.negatives_per_caption == 1
    left = asdict(independent)
    right = asdict(shared)
    for value in (left, right):
        value.pop("source_path")
        value["auxiliary"].pop("arm")
        value["auxiliary"].pop("support_mode")
        value["output"].pop("root")
    assert left == right


def test_reliability_gate_fails_closed_until_instability_evidence(
    tmp_path: Path,
) -> None:
    value = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    value["auxiliary"]["reliability_gate"] = True
    path = tmp_path / "premature_gate.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="deferred"):
        load_config(path)
