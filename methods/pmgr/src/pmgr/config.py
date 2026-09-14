from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


_SCHEMA: dict[str, set[str]] = {
    "": {
        "spec_version", "upstream_commit", "dataset", "seed", "paths", "initialization_mode",
        "data", "scoring", "loss", "training", "engine", "validation",
    },
    "paths": {
        "train_manifest", "validation_manifest", "test_manifest", "train_index",
        "validation_index", "test_index", "baseline_resolved_args", "initialization_checkpoint",
        "baseline_resolved_args_sha256", "initialization_checkpoint_sha256", "output_dir",
        "cico_root",
    },
    "data": {
        "feature_mix", "feature_mix_alpha", "max_features", "max_text_tokens",
        "text_augmentation", "expected_train_groups", "expected_train_videos",
        "expected_validation_groups", "expected_validation_videos",
        "expected_test_groups", "expected_test_videos",
        "drop_incomplete_group_batch",
    },
    "data.text_augmentation": {"enabled", "kind", "probability"},
    "scoring": {
        "mask_policy", "dual_mix", "inner_temperature", "normalize_eps",
        "maximum_contrast_scale",
    },
    "loss": {"mode", "rank_mix", "rank_eta"},
    "training": {
        "effective_groups", "epochs", "optimizer", "learning_rate", "coef_lr",
        "warmup_fraction", "beta1", "beta2", "epsilon", "weight_decay", "clipping_policy",
        "gradient_accumulation_steps", "num_workers",
    },
    "engine": {
        "mode", "distributed", "video_encoder_microbatch", "text_encoder_microbatch",
        "score_video_block", "score_text_block", "precision",
    },
    "validation": {
        "every_epochs", "primary", "secondary", "tie_break", "metric_policy",
        "video_encoder_batch", "text_encoder_batch", "score_video_block", "score_text_block",
    },
}


_LOSS_MODES = {
    "legacy_cico", "single_mixed_ce", "all_uniform_ce", "all_set_ce", "group_ce",
    "all_uniform_ce_population_weighted", "all_set_ce_population_weighted",
    "group_ce_batch_mean", "single_rank", "pmgr", "smooth_ap_single_positive_control",
}
_CE_ONLY_MODES = {
    "legacy_cico", "single_mixed_ce", "all_uniform_ce", "all_set_ce", "group_ce",
    "all_uniform_ce_population_weighted", "all_set_ce_population_weighted",
    "group_ce_batch_mean",
}


def _check_unknown(value: Mapping[str, Any], prefix: str = "") -> None:
    allowed = _SCHEMA[prefix]
    unknown = sorted(set(value) - allowed)
    if unknown:
        name = prefix or "top level"
        raise ConfigError(f"unknown {name} config keys: {', '.join(unknown)}")
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        if child_prefix in _SCHEMA:
            if not isinstance(child, Mapping):
                raise ConfigError(f"{child_prefix} must be an object")
            _check_unknown(child, child_prefix)


def _required(config: Mapping[str, Any], section: str, names: set[str]) -> None:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} must be an object")
    missing = sorted(name for name in names if name not in value)
    if missing:
        raise ConfigError(f"{section} lacks required keys: {', '.join(missing)}")


def validate_config(config: Mapping[str, Any], *, mode: str = "audit") -> None:
    if mode not in {"audit", "train", "validation", "final_test"}:
        raise ConfigError(f"unsupported config validation mode: {mode}")
    _check_unknown(config)
    for section in ("paths", "data", "scoring", "loss", "training", "engine", "validation"):
        _required(config, section, _SCHEMA[section])
    if config.get("spec_version") != "pmgr-1.0":
        raise ConfigError("spec_version must be pmgr-1.0")
    commit = str(config.get("upstream_commit", ""))
    if len(commit) != 40:
        raise ConfigError("upstream_commit must be a full Git revision")
    if config.get("initialization_mode") not in {"weights_only_continuation", "full_reproduction"}:
        raise ConfigError("unsupported initialization_mode")
    data = config["data"]
    if data["feature_mix"] != "sum" or float(data["feature_mix_alpha"]) != 0.8:
        raise ConfigError("PMGR matched arms require the reproduced sum mixture with alpha=0.8")
    if int(data["max_features"]) < 1 or int(data["max_text_tokens"]) < 2:
        raise ConfigError("feature/text lengths must be positive")
    augmentation = data["text_augmentation"]
    if augmentation["kind"] != "random_swap" or not 0 <= float(augmentation["probability"]) <= 1:
        raise ConfigError("text augmentation must use the declared random_swap policy")
    for prefix in ("train", "validation"):
        groups = int(data[f"expected_{prefix}_groups"])
        videos = int(data[f"expected_{prefix}_videos"])
        if groups < 1 or videos < groups:
            raise ConfigError(f"invalid expected {prefix} population counts")
    test_counts = (data["expected_test_groups"], data["expected_test_videos"])
    if (test_counts[0] is None) != (test_counts[1] is None):
        raise ConfigError("test group/video counts must be both null or both populated")
    if test_counts[0] is not None:
        test_groups, test_videos = map(int, test_counts)
        if test_groups < 1 or test_videos < test_groups:
            raise ConfigError("invalid expected test population counts")
    scoring = config["scoring"]
    if scoring["mask_policy"] not in {"valid_tokens_only", "legacy_unmasked"}:
        raise ConfigError("unsupported mask policy")
    if not 0 <= float(scoring["dual_mix"]) <= 1 or float(scoring["inner_temperature"]) <= 0:
        raise ConfigError("invalid scorer constants")
    if (
        float(scoring["dual_mix"]) != 0.5
        or float(scoring["inner_temperature"]) != 0.07
        or float(scoring["maximum_contrast_scale"]) != 100.0
    ):
        raise ConfigError("matched PMGR arms require dual_mix=0.5, sigma=0.07, and scale cap=100")
    loss = config["loss"]
    if loss["mode"] not in _LOSS_MODES or not 0 <= float(loss["rank_mix"]) <= 1:
        raise ConfigError("invalid loss mode or rank mixture")
    if float(loss["rank_eta"]) <= 0:
        raise ConfigError("rank_eta must be positive")
    rank_mix = float(loss["rank_mix"])
    if loss["mode"] in _CE_ONLY_MODES and rank_mix != 0:
        raise ConfigError(f"{loss['mode']} requires rank_mix=0")
    if loss["mode"] == "single_rank" and rank_mix != 1:
        raise ConfigError("single_rank requires rank_mix=1 as an explicit pure-rank declaration")
    if loss["mode"] in {"pmgr", "smooth_ap_single_positive_control"} and rank_mix == 0:
        raise ConfigError(f"{loss['mode']} requires a nonzero rank_mix")
    training = config["training"]
    if training["optimizer"] != "inherited_BertAdam" or training["clipping_policy"] != "legacy_global_and_parameter":
        raise ConfigError("initial PMGR runs must preserve the inherited optimizer/clipping policy")
    exact_optimizer = {
        "learning_rate": 1e-5,
        "coef_lr": 1.0,
        "warmup_fraction": 0.1,
        "beta1": 0.9,
        "beta2": 0.98,
        "epsilon": 1e-6,
        "weight_decay": 0.001,
    }
    changed_optimizer = [
        name for name, expected in exact_optimizer.items() if float(training[name]) != expected
    ]
    if changed_optimizer:
        raise ConfigError(
            "matched PMGR arms changed inherited optimizer fields: "
            + ", ".join(changed_optimizer)
        )
    for name in ("effective_groups", "epochs", "gradient_accumulation_steps", "num_workers"):
        minimum = 2 if name == "effective_groups" else 1
        if int(training[name]) < minimum:
            raise ConfigError(f"training.{name} must be >= {minimum}")
    if int(training["gradient_accumulation_steps"]) != 1:
        raise ConfigError("PMGR effective batches are not independent gradient-accumulation batches")
    engine = config["engine"]
    if engine["mode"] not in {"direct", "two_level_replay"} or engine["distributed"] not in {"none", "manual_sum_replay"}:
        raise ConfigError("unsupported training engine")
    if engine["distributed"] == "manual_sum_replay" and engine["mode"] != "two_level_replay":
        raise ConfigError("manual_sum_replay requires the two_level_replay engine")
    if engine["precision"] != "fp32":
        raise ConfigError("PMGR first implementation requires FP32 encoder replay")
    for name in (
        "video_encoder_microbatch", "text_encoder_microbatch", "score_video_block",
        "score_text_block",
    ):
        if int(engine[name]) < 1:
            raise ConfigError(f"engine.{name} must be positive")
    validation = config["validation"]
    if (
        validation["metric_policy"] != "stable_candidate_order_v1"
        or validation["primary"] != "mean_t2v_v2t_r1"
        or validation["secondary"] != "mean_bidirectional_r5_r10"
        or validation["tie_break"] != "earliest_epoch"
    ):
        raise ConfigError("validation selection must use the declared stable full-gallery policy")
    if int(validation["every_epochs"]) < 1:
        raise ConfigError("validation.every_epochs must be positive")
    for name in ("video_encoder_batch", "text_encoder_batch", "score_video_block", "score_text_block"):
        if int(validation[name]) < 1:
            raise ConfigError(f"validation.{name} must be positive")
    paths = config["paths"]
    test_paths = (paths["test_manifest"], paths["test_index"])
    if (test_paths[0] is None) != (test_paths[1] is None):
        raise ConfigError("test manifest/index must be both null or both populated")
    if mode in {"train", "validation"} and any(path is not None for path in test_paths):
        raise ConfigError(f"test remains locked during {mode}")
    if mode == "final_test" and (any(path is None for path in test_paths) or test_counts[0] is None):
        raise ConfigError("final_test requires explicit test manifest, index, and population counts")
    if mode != "audit":
        required_paths = ("train_index", "validation_index", "baseline_resolved_args", "initialization_checkpoint", "cico_root")
        unresolved = [name for name in required_paths if not paths.get(name)]
        if unresolved:
            raise ConfigError("unresolved training paths: " + ", ".join(unresolved))


def load_config(path: str | Path, *, mode: str = "audit") -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot load config {source}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError("top-level config must be an object")
    validate_config(value, mode=mode)
    return value


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def resolved_with_overrides(
    config: Mapping[str, Any], *, validation_mode: str = "train", **overrides: Any
) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    for dotted, replacement in overrides.items():
        if replacement is None:
            continue
        current = value
        parts = dotted.split("__")
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = replacement
    validate_config(value, mode=validation_mode)
    return value
