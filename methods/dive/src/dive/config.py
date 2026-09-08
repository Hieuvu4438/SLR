from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, TypedDict, cast

import yaml


class ConfigError(ValueError):
    """Raised when a DIVE config violates the implementation contract."""


class RunConfig(TypedDict):
    seed: int
    profile: str
    output_root: str
    variant: str
    comparison_group: str


class DiveConfig(TypedDict):
    spec_version: str
    run: RunConfig
    data: dict[str, Any]
    baseline: dict[str, Any]
    text: dict[str, Any]
    temporal: dict[str, Any]
    evidence: dict[str, Any]
    mining: dict[str, Any]
    support: dict[str, Any]
    sampler: dict[str, Any]
    loss: dict[str, Any]
    train: dict[str, Any]
    selection: dict[str, Any]
    evaluation: dict[str, Any]


_ALLOWED: dict[str, frozenset[str]] = {
    "run": frozenset({"seed", "profile", "output_root", "variant", "comparison_group"}),
    "data": frozenset(
        {
            "dataset", "train_manifest", "dev_manifest", "test_manifest", "train_relations",
            "relevance_dir", "video_root", "pose_root", "rgb_cache_root",
            "translation_artifact", "allow_split_overlap",
            "preparation_protocol", "upstream_root", "train_annotation", "dev_annotation",
            "test_annotation", "train_timing_annotation", "frame_maps_dir",
        }
    ),
    "baseline": frozenset(
        {
            "family", "upstream_commit", "reproduction_config", "initial_weights",
            "locked_checkpoint", "score_branch", "dual_mix", "score_scale",
        }
    ),
    "text": frozenset(
        {
            "tokenizer_artifact", "normalization_version", "unitizer_version", "unit_pool",
            "target_source", "truncate_policy", "reject_partial_targets",
        }
    ),
    "temporal": frozenset(
        {
            "canonical_policy", "clip_steps", "dense_stride_steps", "max_clips",
            "support_view_offsets_steps", "require_distinct_views", "record_raw_frame_map",
            "forbid_unrecorded_global_preprocessing",
        }
    ),
    "evidence": frozenset(
        {
            "output_dim", "hidden_dim", "dropout", "freeze_bn_statistics", "train_bn_affine",
            "normalize_epsilon", "tau_alignment", "reference_cache_dtype", "teacher_update",
        }
    ),
    "mining": frozenset(
        {
            "shortlist_per_direction", "rerank_topk", "hardness", "semantic_rule_set",
            "schema_audit_artifact", "shortlist_audit_queries", "shortlist_audit_seed",
        }
    ),
    "support": frozenset(
        {
            "h_min", "mass_epsilon", "tau_localization", "eta_positive", "eta_differential",
            "min_stability", "target_mass", "max_clip_fraction", "min_retained_mass",
            "max_raw_rf_fraction", "strict_schema_weight", "audited_lexical_weight",
            "failed_record_policy",
        }
    ),
    "sampler": frozenset(
        {
            "effective_batch_size", "contrasts_per_step", "endpoint_quota_per_epoch",
            "unique_sample_ids", "preserve_pair_schedule_across_controls",
        }
    ),
    "loss": frozenset(
        {
            "tau_retrieval", "tau_pair", "tau_local_margin", "local_margin_alpha",
            "pair_target_margin", "lambda_pair", "lambda_local", "lambda_preservation",
            "auxiliary_denominator", "pair_feasibility_bound",
        }
    ),
    "train": frozenset(
        {
            "warmup_epochs", "pilot_epochs", "full_epochs", "budget_mode", "gamma_train",
            "optimizer", "lr_projector", "lr_pose", "weight_decay", "scheduler",
            "warmup_fraction", "minimum_lr_fraction", "grad_clip_norm", "amp", "world_size",
            "gradient_accumulation_steps", "checkpoint_every_epoch", "include_initial_checkpoint",
        }
    ),
    "selection": frozenset(
        {
            "split", "metric", "checkpoint_gamma", "checkpoint_tie_break", "gamma_grid",
            "gamma_tie_break",
        }
    ),
    "evaluation": frozenset(
        {
            "gallery", "query_chunk", "candidate_chunk", "topk", "score_layout", "ties",
            "bootstrap_replicates", "bootstrap_seed", "final_seeds", "final_experiment_plan",
        }
    ),
}

_TOP_LEVEL = frozenset({"spec_version", "extends", *_ALLOWED})
_MAIN_VARIANTS = frozenset({"A0", "A1", "A2", "A2h", "A3", "A4", "A5", "A6"})


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ConfigError("cyclic config inheritance: " + " -> ".join(map(str, (*stack, path))))
    if not path.is_file():
        raise ConfigError(f"config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"top-level YAML must be a mapping: {path}")
    raw = dict(raw)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parents = [parent] if isinstance(parent, str) else parent
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ConfigError("extends must be a path or list of paths")
    merged: dict[str, Any] = {}
    for parent_name in parents:
        merged = _merge(merged, _load(path.parent / parent_name, (*stack, path)))
    return _merge(merged, raw)


def load_config(path: str | Path, *, validate: bool = True) -> DiveConfig:
    config = _load(Path(path), ())
    if validate:
        validate_config(config)
    return cast(DiveConfig, config)


def canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def dump_resolved(config: Mapping[str, Any], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True), encoding="utf-8")
    return config_hash(config)


def _require_mapping(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    return value


def _number(section: Mapping[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"expected numeric value for {key}")
    return float(value)


def validate_config(config: Mapping[str, Any]) -> None:
    unknown_top = sorted(set(config) - _TOP_LEVEL)
    if unknown_top:
        raise ConfigError("unknown top-level config keys: " + ", ".join(unknown_top))
    if config.get("spec_version") != "dive_v2_impl_1_0":
        raise ConfigError("spec_version must be dive_v2_impl_1_0")
    missing_sections = sorted(set(_ALLOWED) - set(config))
    if missing_sections:
        raise ConfigError("missing config sections: " + ", ".join(missing_sections))
    for name, allowed in _ALLOWED.items():
        section = _require_mapping(config, name)
        unknown = sorted(set(section) - allowed)
        if unknown:
            raise ConfigError(f"unknown keys in {name}: " + ", ".join(unknown))
        missing = sorted(allowed - set(section))
        if missing:
            raise ConfigError(f"missing keys in {name}: " + ", ".join(missing))

    run = _require_mapping(config, "run")
    evidence = _require_mapping(config, "evidence")
    temporal = _require_mapping(config, "temporal")
    support = _require_mapping(config, "support")
    sampler = _require_mapping(config, "sampler")
    loss = _require_mapping(config, "loss")
    train = _require_mapping(config, "train")
    selection = _require_mapping(config, "selection")
    evaluation = _require_mapping(config, "evaluation")

    if run.get("variant") not in _MAIN_VARIANTS and not str(run.get("variant", "")).startswith("B"):
        raise ConfigError(f"unsupported run.variant: {run.get('variant')}")
    if run.get("profile") not in {"correctness", "optimization", "fixture"}:
        raise ConfigError("run.profile must be correctness, optimization, or fixture")
    for name, section, key in (
        ("alignment", evidence, "tau_alignment"),
        ("localization", support, "tau_localization"),
        ("retrieval", loss, "tau_retrieval"),
        ("pair", loss, "tau_pair"),
        ("local margin", loss, "tau_local_margin"),
    ):
        if _number(section, key) <= 0:
            raise ConfigError(f"{name} temperature must be positive")
    if _number(train, "gamma_train") <= 0 and run.get("variant") not in {"A0", "A1"}:
        raise ConfigError("train.gamma_train must be positive for centered-residual training")
    if int(evidence.get("output_dim", 0)) != 512:
        raise ConfigError("evidence.output_dim must equal the frozen text dimension 512")
    if int(evidence.get("hidden_dim", 0)) != 1024:
        raise ConfigError("evidence.hidden_dim must be 1024 in the main profile")
    if _number(evidence, "dropout") != 0:
        raise ConfigError("evidence.dropout must be zero")
    if evidence.get("teacher_update") != "frozen_after_warmup":
        raise ConfigError("evidence teacher must be frozen after warm-up")
    if not evidence.get("freeze_bn_statistics") or evidence.get("train_bn_affine"):
        raise ConfigError("main evidence profile requires frozen BN statistics and affine")
    if _number(loss, "pair_target_margin") != 0:
        raise ConfigError("main pair_target_margin must be zero")
    fractions = (
        (support, "min_stability"), (support, "target_mass"),
        (support, "max_clip_fraction"), (support, "min_retained_mass"),
        (support, "max_raw_rf_fraction"), (train, "warmup_fraction"),
        (train, "minimum_lr_fraction"),
    )
    for section, key in fractions:
        value = _number(section, key)
        if not 0 <= value <= 1:
            raise ConfigError(f"{key} must be in [0, 1]")
    if _number(support, "min_retained_mass") > _number(support, "target_mass"):
        raise ConfigError("support.min_retained_mass cannot exceed target_mass")
    if int(temporal.get("clip_steps", 0)) <= 0 or int(temporal.get("dense_stride_steps", 0)) <= 0:
        raise ConfigError("temporal clip steps and stride must be positive")
    if int(temporal.get("max_clips", 0)) < 1:
        raise ConfigError("temporal.max_clips must be positive")
    offsets = temporal.get("support_view_offsets_steps")
    if temporal.get("require_distinct_views") and (not isinstance(offsets, list) or len(set(offsets)) < 2):
        raise ConfigError("main support requires at least two distinct view offsets")
    batch_size = int(sampler.get("effective_batch_size", 0))
    contrasts = int(sampler.get("contrasts_per_step", 0))
    if batch_size <= 0 or contrasts < 0 or 2 * contrasts > batch_size:
        raise ConfigError("sampler requires 2*contrasts_per_step <= effective_batch_size")
    if selection.get("split") != "dev":
        raise ConfigError("selection split must be dev, never test")
    gamma_grid = selection.get("gamma_grid")
    if not isinstance(gamma_grid, list) or not gamma_grid or any(float(x) < 0 for x in gamma_grid):
        raise ConfigError("selection.gamma_grid must be a non-empty nonnegative list")
    if evaluation.get("gallery") != "full":
        raise ConfigError("standard evaluation requires the full gallery")
    if evaluation.get("score_layout") != "video_rows_text_columns":
        raise ConfigError("score layout must have video rows and text columns")
    if evaluation.get("ties") != "score_desc_then_candidate_id":
        raise ConfigError("tie policy must be score descending then candidate ID")
    if bool(train.get("amp")) and run.get("profile") == "correctness":
        raise ConfigError("AMP is not allowed in the FP32 correctness profile")
    if int(train.get("world_size", 1)) != 1 and run.get("profile") == "correctness":
        raise ConfigError("correctness profile is single-process")
    if train.get("optimizer") != "adamw":
        raise ConfigError("main optimizer must be AdamW")


def required_resource_paths(config: Mapping[str, Any], stage: str) -> dict[str, str | None]:
    """Return only prerequisites for a stage; null paths remain explicit blockers."""
    data = _require_mapping(config, "data")
    baseline = _require_mapping(config, "baseline")
    text = _require_mapping(config, "text")
    mining = _require_mapping(config, "mining")
    common_data = {
        "data.video_root": data.get("video_root"),
        "data.pose_root": data.get("pose_root"),
    }
    requirements: dict[str, dict[str, str | None]] = {
        "fixture": {},
        "prepare": {
            **common_data,
            "data.upstream_root": data.get("upstream_root"),
            "data.train_annotation": data.get("train_annotation"),
            "data.train_timing_annotation": data.get("train_timing_annotation"),
            "data.dev_annotation": data.get("dev_annotation"),
            "data.test_annotation": data.get("test_annotation"),
        },
        "validate_data": {
            **common_data,
            "data.train_manifest": data.get("train_manifest"),
            "data.dev_manifest": data.get("dev_manifest"),
            "data.test_manifest": data.get("test_manifest"),
            "data.train_relations": data.get("train_relations"),
            "data.relevance_dir": data.get("relevance_dir"),
            "data.rgb_cache_root": data.get("rgb_cache_root"),
            "data.frame_maps_dir": data.get("frame_maps_dir"),
        },
        "baseline_train": {
            **common_data,
            "data.train_manifest": data.get("train_manifest"),
            "data.dev_manifest": data.get("dev_manifest"),
            "baseline.reproduction_config": baseline.get("reproduction_config"),
            "baseline.initial_weights": baseline.get("initial_weights"),
        },
        "baseline_validate": {
            "data.dev_manifest": data.get("dev_manifest"),
            "baseline.locked_checkpoint": baseline.get("locked_checkpoint"),
        },
        "warmup": {
            "data.train_manifest": data.get("train_manifest"),
            "data.dev_manifest": data.get("dev_manifest"),
            "baseline.locked_checkpoint": baseline.get("locked_checkpoint"),
            "text.tokenizer_artifact": text.get("tokenizer_artifact"),
        },
        "mine_finalize": {
            "data.train_manifest": data.get("train_manifest"),
            "data.train_relations": data.get("train_relations"),
            "mining.schema_audit_artifact": mining.get("schema_audit_artifact"),
        },
        "evaluate_test": {
            "data.test_manifest": data.get("test_manifest"),
            "data.relevance_dir": data.get("relevance_dir"),
            "evaluation.final_experiment_plan": _require_mapping(config, "evaluation").get(
                "final_experiment_plan"
            ),
        },
    }
    if stage not in requirements:
        raise ConfigError(f"unknown doctor stage: {stage}")
    return requirements[stage]
