from __future__ import annotations

from pathlib import Path

import pytest

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
