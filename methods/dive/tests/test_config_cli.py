from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dive.cli import main
from dive.config import ConfigError, config_hash, load_config, validate_config


HERE = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = HERE / "configs" / "fixture.yaml"


def test_fixture_config_is_strict_and_hash_is_canonical():
    config = load_config(FIXTURE_CONFIG)
    assert config["run"]["profile"] == "fixture"
    assert config_hash(config) == config_hash(dict(reversed(list(config.items()))))
    broken = json.loads(json.dumps(config))
    broken["evidence"]["tau_alignmnt"] = broken["evidence"].pop("tau_alignment")
    with pytest.raises(ConfigError, match="unknown keys in evidence"):
        validate_config(broken)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("selection", "split"), "test", "selection split"),
        (("train", "gamma_train"), 0, "gamma_train"),
        (("support", "min_retained_mass"), 0.7, "cannot exceed"),
        (("sampler", "contrasts_per_step"), 5, "2\\*contrasts"),
    ],
)
def test_semantic_validation_rejects_protocol_violations(path, value, message):
    config = load_config(FIXTURE_CONFIG)
    config[path[0]][path[1]] = value
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_doctor_fixture_writes_machine_readable_report(tmp_path, capsys):
    output = tmp_path / "doctor.json"
    result = main(["doctor", "--config", str(FIXTURE_CONFIG), "--stage", "fixture", "--output", str(output)])
    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ready"] is True
    assert report["environment"]["torch"]["installed"] is True
    assert json.loads(capsys.readouterr().out)["schema_version"] == "doctor.v1"


def test_doctor_fails_closed_for_missing_stage_resource(tmp_path):
    config = load_config(FIXTURE_CONFIG)
    config["data"]["dev_manifest"] = str(tmp_path / "missing.jsonl")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "doctor.json"
    result = main(
        ["doctor", "--config", str(path), "--stage", "baseline_validate", "--output", str(output)]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 2
    assert {item["error_code"] for item in report["missing"]} == {
        "MISSING_DEV_ARTIFACT",
        "MISSING_BASELINE_CHECKPOINT",
    }
