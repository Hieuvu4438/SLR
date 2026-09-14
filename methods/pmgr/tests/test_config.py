from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmgr.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[3]


def _config():
    return json.loads((ROOT / "methods/pmgr/configs/pmgr_csl.json").read_text())


def test_checked_config_is_strict_and_valid(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = load_config(ROOT / "methods/pmgr/configs/pmgr_csl.json", mode="train")
    assert config["loss"]["mode"] == "group_ce"


def test_unknown_and_contradictory_options_fail():
    config = _config()
    config["mystery"] = True
    with pytest.raises(ConfigError, match="unknown"):
        validate_config(config)
    config = _config()
    config["loss"]["rank_mix"] = 0.25
    with pytest.raises(ConfigError, match="requires rank_mix=0"):
        validate_config(config)
    config = _config()
    config["loss"]["mode"] = "pmgr"
    with pytest.raises(ConfigError, match="nonzero rank_mix"):
        validate_config(config)
    config["loss"]["rank_mix"] = 0.25
    validate_config(config)
    config = _config()
    config["loss"]["mode"] = "single_rank"
    with pytest.raises(ConfigError, match="rank_mix=1"):
        validate_config(config)
    config = _config()
    config["engine"]["mode"] = "direct"
    config["engine"]["distributed"] = "manual_sum_replay"
    with pytest.raises(ConfigError, match="requires"):
        validate_config(config)


def test_training_never_accepts_unlocked_test_paths():
    config = _config()
    config["paths"]["test_index"] = "some/test.json"
    with pytest.raises(ConfigError, match="test remains locked"):
        validate_config(config, mode="train")
