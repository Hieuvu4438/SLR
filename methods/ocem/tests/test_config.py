from __future__ import annotations

import pytest

from ocem.config import ConfigError, load_config


def _write(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_resource_config_rejects_unknown_key(tmp_path) -> None:
    path = _write(
        tmp_path,
        """\
schema_version: ocem.resources.v1
project: {project_root_env: OCEM_PROJECT_ROOT, data_root_env: OCEM_DATA_ROOT, run_root_env: OCEM_RUN_ROOT}
resources:
  - id: oxford_i3d
    kind: checkpoint
    required: true
    local_path: /tmp/model.pt
    access_status: discovered
    surprise: forbidden
""",
    )
    with pytest.raises(ConfigError, match="unknown keys: surprise"):
        load_config(path, kind="resources")


def test_config_rejects_unresolved_environment(tmp_path) -> None:
    path = _write(
        tmp_path,
        """\
schema_version: ocem.protocol.v1
dataset: phoenix2014t
release: v3
resource_group: rgb
official_splits: {train: "${OCEM_TEST_MISSING_ENV}/train.jsonl"}
""",
    )
    with pytest.raises(ConfigError, match="OCEM_TEST_MISSING_ENV"):
        load_config(path, kind="protocol")


def test_config_rejects_nonfinite_float(tmp_path) -> None:
    path = _write(
        tmp_path,
        """\
schema_version: ocem.protocol.v1
dataset: phoenix2014t
release: v3
resource_group: rgb
evaluation: {tie_tolerance: .nan}
""",
    )
    with pytest.raises(ConfigError, match="NaN and infinity"):
        load_config(path, kind="protocol")


def test_experiment_rejects_unknown_nested_key(tmp_path) -> None:
    path = _write(
        tmp_path,
        """\
schema_version: ocem.experiment.v1
run: {name: fixture, phase: minimal_head, seeds: [0], output_root: /tmp}
data: {dataset: phoenix2014t}
baseline: {implementation: cico_adapter}
model: {local_input_dim: 1024, hidden_transformer: forbidden}
score: {kind: independent}
solver: {backend: closed_form}
training: {optimizer: adam}
mining: {source: frozen_reproduced_baseline}
evaluation: {mode: full_gallery}
""",
    )
    with pytest.raises(ConfigError, match="unknown keys: hidden_transformer"):
        load_config(path, kind="experiment")


def test_resource_config_rejects_required_excluded_dependency(tmp_path) -> None:
    path = _write(
        tmp_path,
        """\
schema_version: ocem.resources.v1
project: {project_root_env: OCEM_PROJECT_ROOT, data_root_env: OCEM_DATA_ROOT, run_root_env: OCEM_RUN_ROOT}
resources:
  - id: seds_artifacts
    kind: forbidden
    required: true
    local_path: null
    access_status: unavailable
    excluded_dependency: true
""",
    )
    with pytest.raises(ConfigError, match="excluded dependency cannot be required"):
        load_config(path, kind="resources")
