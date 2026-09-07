from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration violates an ELSC contract."""


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"cyclic config inheritance: {cycle}")
    if not path.is_file():
        raise ConfigError(f"config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"top-level YAML must be a mapping: {path}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parents = [parent] if isinstance(parent, str) else parent
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ConfigError("extends must be a path or a list of paths")
    merged: dict[str, Any] = {}
    for parent_name in parents:
        merged = _merge(merged, _load(path.parent / parent_name, (*stack, path)))
    return _merge(merged, raw)


def load_config(
    path: str | Path, *, validate: bool = True, stage: str | None = None
) -> dict[str, Any]:
    config = _load(Path(path), ())
    if validate:
        validate_config(config, stage=stage)
    return config


def canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def dump_resolved(config: Mapping[str, Any], path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True, allow_unicode=True)
    return config_hash(config)


def _walk_required(value: Any, prefix: str = "") -> list[str]:
    missing: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            missing.extend(_walk_required(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            missing.extend(_walk_required(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and value.lower().startswith("required"):
        missing.append(prefix)
    return missing


def validate_config(config: Mapping[str, Any], *, stage: str | None = None) -> None:
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    train = config.get("train", {})
    if train.get("eval_split", "dev") != "dev":
        raise ConfigError("training checkpoint selection must use the dev split")
    precision = train.get("precision", "fp32")
    if precision not in {"fp32", "amp_fp16", "amp_bf16"}:
        raise ConfigError("train.precision must be fp32, amp_fp16, or amp_bf16")
    evaluation = config.get("evaluation", {})
    if evaluation.get("filter_by_aux_eligibility", False):
        raise ConfigError("evaluation must retain the full gallery")
    adapter = config.get("model", {}).get("adapter", {})
    radius = int(adapter.get("radius", 0))
    data = config.get("data", {})
    if data.get("text_augmentation", "none") not in {"none", "cico_random_swap_v1"}:
        raise ConfigError(
            "data.text_augmentation must be none or cico_random_swap_v1"
        )
    if radius != 0:
        raise ConfigError(
            "only pointwise model.adapter.radius=0 is implemented; dense temporal adapters "
            "require a separate locality implementation"
        )
    evidence = config.get("evidence", {})
    sources = config.get("sources", {})
    if sources.get("require_feature_sidecars", False):
        expected_feature_fields = (
            "feature_agnostic_stream_name",
            "feature_aware_stream_name",
            "feature_agnostic_checkpoint_sha256",
            "feature_aware_checkpoint_sha256",
            "feature_recipe_sha256",
        )
        missing_expected = [key for key in expected_feature_fields if not sources.get(key)]
        if missing_expected:
            raise ConfigError(
                "sidecar validation requires expected feature provenance: "
                + ", ".join(missing_expected)
            )
        if sources.get("require_temporal_metadata", False) and not sources.get(
            "temporal_metadata_root"
        ):
            raise ConfigError(
                "sidecar validation requires temporal_metadata_root when temporal metadata is required"
            )
    if evidence.get("enabled"):
        rf_root = sources.get("temporal_metadata_root")
        if not rf_root or str(rf_root).lower().startswith("required"):
            if stage not in {None, "schema", "assets"}:
                raise ConfigError("ELSC-Full requires verified temporal_metadata_root")
        if not evidence.get("control_same_token_count", False):
            raise ConfigError("ELSC-Full requires control_same_token_count=true")
        if evidence.get("mask_fill", "zero") != "zero":
            raise ConfigError("only evidence.mask_fill=zero is implemented")
        if int(evidence.get("max_pairs_per_video", 1)) < 1:
            raise ConfigError("evidence.max_pairs_per_video must be positive")
    method = config.get("method", "elsc")
    if method not in {"baseline", "elsc", "matched_caption", "local_word_video"}:
        raise ConfigError(f"unsupported method: {method}")
    support_mode = config.get("aux_support_mode", "teacher")
    if support_mode not in {"teacher", "random_matched", "shuffled_lexical"}:
        raise ConfigError(f"unsupported aux_support_mode: {support_mode}")
    if stage not in {None, "schema"}:
        missing = _walk_required(config)
        if stage == "assets":
            # An asset audit reports unresolved paths instead of hiding them.
            return
        if missing:
            raise ConfigError("unresolved required values: " + ", ".join(missing))
    if method != "baseline" and not adapter.get("enabled", False):
        raise ConfigError("auxiliary methods require an enabled adapter")
    cache = config.get("cache", {})
    random_span_tolerance = float(cache.get("random_span_duration_tolerance", 0.10))
    if not 0.0 <= random_span_tolerance <= 1.0:
        raise ConfigError("cache.random_span_duration_tolerance must be in [0,1]")
    negative_language = cache.get("negative_language")
    caption_language = data.get("caption_language")
    if negative_language and negative_language != caption_language:
        raise ConfigError("negative candidate language must match the model caption language")
    if (
        int(train.get("epochs", 1)) < 1
        or int(train.get("per_device_batch", 1)) < 1
        or int(train.get("accumulation_steps", 1)) < 1
    ):
        raise ConfigError("epochs, batch size, and accumulation steps must be positive")
    if int(train.get("gradient_diagnostic_interval", 100)) < 1:
        raise ConfigError("train.gradient_diagnostic_interval must be positive")
    if not 0.0 <= float(train.get("warmup_ratio", 0.0)) <= 1.0:
        raise ConfigError("train.warmup_ratio must be in [0,1]")
    if float(train.get("grad_clip_norm", 1.0)) <= 0:
        raise ConfigError("train.grad_clip_norm must be positive")
    if train.get("optimizer", "adamw") != "adamw":
        raise ConfigError("only train.optimizer=adamw is implemented")
    beta1 = float(train.get("beta1", 0.9))
    beta2 = float(train.get("beta2", 0.98))
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ConfigError("train beta1 and beta2 must be in [0,1)")
    if float(train.get("epsilon", 1e-6)) <= 0 or float(train.get("weight_decay", 0.0)) < 0:
        raise ConfigError("train epsilon must be positive and weight_decay non-negative")
    backbone_trainable = not config.get("model", {}).get("backbone_frozen", True)
    adapter_trainable = adapter.get("enabled", False) and adapter.get("trainable", True)
    head_trainable = method in {"elsc", "local_word_video"}
    if not (backbone_trainable or adapter_trainable or head_trainable):
        raise ConfigError("all trainable parameters are frozen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve and validate an ELSC YAML config")
    parser.add_argument("config")
    parser.add_argument("--output")
    parser.add_argument("--stage", default="schema")
    args = parser.parse_args(argv)
    config = load_config(args.config, stage=args.stage)
    digest = config_hash(config)
    if args.output:
        dump_resolved(config, args.output)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
