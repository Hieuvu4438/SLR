"""Strict YAML configuration loading for OCEM.

The validator intentionally rejects unknown keys and unresolved environment
variables. Individual work packages extend these schemas as their executable
contracts are implemented; callers must not silently ignore future fields.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration violates an OCEM contract."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_SCHEMAS: dict[str, dict[str, object]] = {
    "resources": {
        "schema_version": None,
        "project": {
            "project_root_env": None,
            "data_root_env": None,
            "run_root_env": None,
        },
        "resources": [
            {
                "id": None,
                "kind": None,
                "required": None,
                "source_url": None,
                "local_path": None,
                "expected_sha256": None,
                "expected_bytes": None,
                "access_status": None,
                "excluded_dependency": None,
                "loader": None,
                "notes": None,
            }
        ],
    },
    "protocol": {
        "schema_version": None,
        "dataset": None,
        "release": None,
        "resource_group": None,
        "official_splits": {
            "train": None,
            "validation": None,
            "test": None,
        },
        "galleries": {
            "t2v": None,
            "v2t": None,
            "keep_official_gallery": None,
        },
        "preprocessing": {
            "fps_policy": None,
            "window_frames": None,
            "window_stride": None,
            "resize": None,
            "crop": None,
            "pixel_range": None,
            "mean": None,
            "std": None,
            "padding_policy": None,
        },
        "baseline": {
            "implementation": None,
            "commit": None,
            "batch_size": None,
            "score_contract": None,
            "reported_t2v_r1": None,
            "reported_v2t_r1": None,
            "reproduction_tolerance_pp": None,
        },
        "evaluation": {
            "full_gallery": None,
            "tie_policy": None,
            "metrics": None,
        },
        "resources": None,
    },
    "experiment": {
        "schema_version": None,
        "run": {"name": None, "phase": None, "seeds": None, "output_root": None},
        "data": {
            "dataset": None,
            "root": None,
            "protocol_lock": None,
            "feature_lock": None,
            "max_video_tokens": None,
            "max_text_input_tokens": None,
            "keep_official_gallery": None,
        },
        "baseline": {
            "implementation": None,
            "reproduction_gate": None,
            "checkpoint": None,
            "freeze_context": None,
            "freeze_logit_scale": None,
            "raw_score_contract": None,
        },
        "model": {
            "local_input_dim": None,
            "embedding_dim": None,
            "local_head": None,
            "i3d_frozen": None,
            "add_pose": None,
            "add_gloss": None,
            "add_synthetic_captions": None,
        },
        "score": {
            "kind": None,
            "geometry": None,
            "reference_mass": None,
            "null_index": None,
            "epsilon": None,
            "null_prior": None,
            "kappa": None,
            "gamma": None,
            "center_zero_value": None,
        },
        "solver": {
            "backend": None,
            "dtype": None,
            "pair_block_size": None,
            "retry_iterations": None,
            "train_phi_gap": None,
            "eval_phi_gap": None,
            "train_capacity_residual": None,
            "eval_capacity_residual": None,
            "allow_uncertified_scores": None,
            "cpu_reference_fallback": None,
        },
        "training": {
            "optimizer": None,
            "new_head_lr": None,
            "contextual_lr": None,
            "lambda_mix": None,
            "global_contrastive_batch": None,
            "epochs": None,
            "projection_warmup_epochs": None,
            "encoder_precision": None,
            "schedule_from_protocol_lock": None,
            "use_gradient_accumulation_as_extra_negatives": None,
            "exact_duplicate_negative_mask": None,
        },
        "mining": {
            "source": None,
            "hard_per_anchor": None,
            "random_per_anchor": None,
            "row_score": None,
            "column_score": None,
            "uniform_during_warmup": None,
            "union_for_compute_only": None,
        },
        "evaluation": {
            "mode": None,
            "tie_policy": None,
            "selection_metric": None,
            "test_for_selection": None,
            "report_directions": None,
            "report_recall_at": None,
        },
    },
}


def _expand_environment(value: Any, path: str) -> Any:
    if isinstance(value, str):
        missing = sorted({name for name in _ENV_PATTERN.findall(value) if name not in os.environ})
        if missing:
            raise ConfigError(f"{path}: unresolved environment variables: {', '.join(missing)}")
        return _ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], value)
    if isinstance(value, list):
        return [_expand_environment(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {
            key: _expand_environment(item, f"{path}.{key}")
            for key, item in value.items()
        }
    return value


def _reject_nonfinite(value: Any, path: str = "config") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{path}: NaN and infinity are forbidden")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")


def _check_unknown_keys(value: Any, schema: object, path: str) -> None:
    if schema is None:
        return
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        for index, item in enumerate(value):
            _check_unknown_keys(item, schema[0], f"{path}[{index}]")
        return
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping")
    assert isinstance(schema, dict)
    unknown = sorted(set(value) - set(schema))
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {', '.join(unknown)}")
    for key, item in value.items():
        _check_unknown_keys(item, schema[key], f"{path}.{key}")


def _validate_common(config: Mapping[str, Any], kind: str) -> None:
    expected_versions = {
        "resources": "ocem.resources.v1",
        "protocol": "ocem.protocol.v1",
        "experiment": "ocem.experiment.v1",
    }
    expected = expected_versions[kind]
    if config.get("schema_version") != expected:
        raise ConfigError(f"config.schema_version: expected {expected!r}")


def _validate_resources(config: Mapping[str, Any]) -> None:
    resources = config.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ConfigError("config.resources: expected a non-empty list")
    ids: set[str] = set()
    for index, resource in enumerate(resources):
        path = f"config.resources[{index}]"
        if not isinstance(resource, Mapping):
            raise ConfigError(f"{path}: expected a mapping")
        required = {"id", "kind", "required", "local_path", "access_status"}
        missing = sorted(required - set(resource))
        if missing:
            raise ConfigError(f"{path}: missing keys: {', '.join(missing)}")
        resource_id = resource["id"]
        if not isinstance(resource_id, str) or not resource_id:
            raise ConfigError(f"{path}.id: expected a non-empty string")
        if resource_id in ids:
            raise ConfigError(f"{path}.id: duplicate resource id {resource_id!r}")
        ids.add(resource_id)
        if resource.get("excluded_dependency") and resource.get("required"):
            raise ConfigError(f"{path}: excluded dependency cannot be required")


def _validate_experiment(config: Mapping[str, Any]) -> None:
    required = {"run", "data", "baseline", "model", "score", "solver", "training", "mining", "evaluation"}
    missing = sorted(required - set(config))
    if missing:
        raise ConfigError(f"config: missing keys: {', '.join(missing)}")
    score = config.get("score")
    if not isinstance(score, Mapping):
        raise ConfigError("config.score: expected a mapping")
    if score.get("kind") == "ocem":
        if score.get("geometry") != "actual_window_support_v1":
            raise ConfigError("config.score.geometry: OCEM requires actual_window_support_v1")
        if score.get("center_zero_value") is not True:
            raise ConfigError("config.score.center_zero_value: OCEM requires centering")
        for key in ("epsilon", "null_prior", "kappa"):
            if key not in score:
                raise ConfigError(f"config.score.{key}: required for OCEM")
        if not 0 < float(score["null_prior"]) < 1:
            raise ConfigError("config.score.null_prior: expected 0 < value < 1")
        if float(score["epsilon"]) <= 0 or float(score["kappa"]) <= 0:
            raise ConfigError("config.score: epsilon and kappa must be positive")


def load_config(path: str | Path, *, kind: str, expand_environment: bool = True) -> dict[str, Any]:
    """Load and validate an OCEM YAML config.

    This performs structural validation only. Resource existence, hashes, model
    loading, and prerequisite gates belong to their explicit runtime commands.
    """

    if kind not in _SCHEMAS:
        raise ConfigError(f"unsupported config kind {kind!r}")
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigError("config: expected a YAML mapping")
    _reject_nonfinite(loaded)
    _check_unknown_keys(loaded, _SCHEMAS[kind], "config")
    resolved = _expand_environment(loaded, "config") if expand_environment else loaded
    _validate_common(resolved, kind)
    if kind == "resources":
        _validate_resources(resolved)
    elif kind == "experiment":
        _validate_experiment(resolved)
    return dict(resolved)
