from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_type_hints

import yaml

from .utils import sha256_json


UPRET_COMMIT = "046366227417e1d8ec14145965403462df345984"
SUPPORTED_DATASETS = {"ph", "h2", "csl"}
SUPPORTED_ARMS = {
    "base_initial",
    "base_continuation",
    "span_independent",
    "span_shared",
    "span_random_support",
    "caption_hn",
    "fsc_local",
    "fsc_local_caption_hn",
}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    language: str
    source_annotations: dict[str, str]
    official_membership_json: str
    manifest_dir: str
    agnostic_root: str
    aware_root: str
    agnostic_weight: float
    official_split_annotations: dict[str, str] = field(default_factory=dict)
    combine_type: str = "sum"
    feature_dim: int = 1024
    feature_len: int = 64
    feature_sampling: str = "legacy_linspace"
    text_max_positions: int = 32
    text_overlength_policy: str = "legacy_uniform_bpe"
    auxiliary_full_caption_only: bool = True
    group_member_sampling: str = "stateless_uniform"
    baseline_text_augmentation: str = "random_swap"
    baseline_text_augmentation_probability: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    pretrained_clip_name: str
    clip_checkpoint_path: str
    bpe_path: str
    visual_layers: int = 12
    text_layers: int = 12
    embedding_dim: int = 512
    linear_patch: str = "2d"
    sim_header: str = "Filip"
    loose_type: bool = True
    freeze_layer_num: int = 0
    inner_similarity_temperature: float = 0.07
    dual_mix: float = 0.5
    mix_design: str = "balance"
    distribution_samples: int = 2
    sample_ot_epsilon: float = 0.1
    sample_ot_max_iterations: int = 100
    sample_ot_logit_weight: float = 1.0


@dataclass(frozen=True)
class ReferenceConfig:
    mode: str
    checkpoint: str
    cache_dir: str
    cache_dtype: str = "float32"
    require_exact_hashes: bool = True


@dataclass(frozen=True)
class MinerConfig:
    version: str = "visual_prototype_v1"
    edits_per_negative: int = 1
    retained_occurrence_fraction: float = 0.5
    minimum_concentration: float = 0.000001
    minimum_retained_occurrences: int = 10
    minimum_distinct_groups: int = 5
    candidate_cosine_minimum: float = 0.7
    candidates_per_word: int = 20
    cached_negatives_per_caption: int = 20
    reject_existing_train_caption: bool = True
    reject_word_already_in_caption: bool = True


@dataclass(frozen=True)
class AuxiliaryConfig:
    arm: str
    support_mode: str
    negatives_per_caption: int = 5
    edits_per_negative: int = 1
    tau_support: float = 0.07
    margin: float = 0.05
    aux_weight: float = 0.10
    normalize_text_difference: bool = False
    normalize_support_pool: bool = False
    reliability_gate: bool = False
    caption_loss_weight: float = 0.4
    fsc_loss_weight: float = 0.4
    fsc_loss_name: str = "focal_loss"
    fsc_focal_gamma: float = 1.0
    fsc_label_smoothing: float = 0.1
    fsc_similarity_normalizer: str = "minmax"


@dataclass(frozen=True)
class TrainingConfig:
    base_epochs: int = 200
    finetune_epochs: int = 20
    global_contrastive_batch: int = 512
    gradient_accumulation_steps: int = 1
    optimizer: str = "upstream_bertadam"
    learning_rate: float = 0.00001
    betas: tuple[float, float] = (0.9, 0.98)
    epsilon: float = 0.000001
    weight_decay: float = 0.001
    warmup_fraction: float = 0.1
    schedule: str = "upstream_warmup_cosine"
    restart_optimizer_for_finetuning: bool = True
    max_grad_norm: float = 1.0
    mixed_precision: str = "none"
    deterministic_debug: bool = False
    autograd_global_gather: bool = True
    replicated_distribution_rng: bool = True
    train_video_pair_block: int = 32
    train_text_pair_block: int = 64
    checkpoint_deterministic_score_blocks: bool = True
    num_workers: int = 4
    checkpoint_every_steps: int = 100


@dataclass(frozen=True)
class EvaluationConfig:
    encode_batch_size: int = 64
    complete_candidate_pool: bool = True
    group_aggregation: str = "max"
    tie_policy: str = "stable_manifest_order"
    selection_metric: str = "mean_bidirectional_r1"
    tie_break_metric: str = "mean_bidirectional_r5"
    report_ranks: bool = True
    report_top_k: int = 10
    test_during_training: bool = False


@dataclass(frozen=True)
class OutputConfig:
    root: str
    save_last: bool = True
    save_best_dev: bool = True
    atomic_checkpoints: bool = True


@dataclass(frozen=True)
class Method1Config:
    schema_version: int
    method_version: str
    baseline_version: str
    upstream_commit: str
    seed: int
    data: DataConfig
    model: ModelConfig
    reference: ReferenceConfig
    miner: MinerConfig
    auxiliary: AuxiliaryConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    source_path: str = field(default="", compare=False)

    @property
    def digest(self) -> str:
        value = asdict(self)
        value.pop("source_path", None)
        return sha256_json(value)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError("schema_version must be 1")
        if self.method_version != "sssc_v1":
            raise ConfigError("method_version must be sssc_v1")
        if self.baseline_version != "upret_main_corrected_v1":
            raise ConfigError("baseline_version must be upret_main_corrected_v1")
        if self.upstream_commit != UPRET_COMMIT:
            raise ConfigError(f"upstream_commit must be pinned to {UPRET_COMMIT}")
        if self.data.dataset not in SUPPORTED_DATASETS:
            raise ConfigError(f"unsupported dataset: {self.data.dataset}")
        if set(self.data.source_annotations) != {"train", "dev", "test"}:
            raise ConfigError("source_annotations must contain exactly train/dev/test")
        if self.data.official_split_annotations and set(self.data.official_split_annotations) != {
            "train",
            "dev",
            "test",
        }:
            raise ConfigError("official_split_annotations must be empty or contain train/dev/test")
        if self.data.combine_type != "sum":
            raise ConfigError("first implementation requires combine_type=sum")
        expected_weight = 0.9 if self.data.dataset == "ph" else 0.8
        if self.data.agnostic_weight != expected_weight:
            raise ConfigError(
                f"{self.data.dataset} requires agnostic_weight={expected_weight} in the fixed implementation"
            )
        if (self.data.feature_dim, self.data.feature_len, self.data.text_max_positions) != (
            1024,
            64,
            32,
        ):
            raise ConfigError("fixed tensor contract requires feature_dim=1024, feature_len=64, text_max_positions=32")
        if self.model.embedding_dim != 512 or self.model.sim_header != "Filip":
            raise ConfigError("fixed baseline requires embedding_dim=512 and sim_header=Filip")
        if not isinstance(self.model.loose_type, bool):
            raise ConfigError("loose_type must be a YAML boolean")
        if self.model.distribution_samples != 2:
            raise ConfigError("corrected baseline preserves two distribution samples")
        if self.auxiliary.arm not in SUPPORTED_ARMS:
            raise ConfigError(f"unsupported auxiliary arm: {self.auxiliary.arm}")
        expected_mode = {
            "span_shared": "shared",
            "span_independent": "independent",
            "span_random_support": "random",
        }.get(self.auxiliary.arm)
        if expected_mode is not None and self.auxiliary.support_mode != expected_mode:
            raise ConfigError(
                f"arm={self.auxiliary.arm} requires support_mode={expected_mode}"
            )
        if self.auxiliary.reliability_gate:
            raise ConfigError(
                "reliability gate is deferred until the ungated support-instability gate"
            )
        if self.auxiliary.edits_per_negative != 1 or self.miner.edits_per_negative != 1:
            raise ConfigError("version 1 supports exactly one lexical edit per negative")
        if self.auxiliary.negatives_per_caption < 1:
            raise ConfigError("negatives_per_caption must be positive")
        if self.auxiliary.negatives_per_caption > self.miner.cached_negatives_per_caption:
            raise ConfigError("K cannot exceed cached_negatives_per_caption")
        if self.auxiliary.normalize_text_difference or self.auxiliary.normalize_support_pool:
            raise ConfigError("the fixed shared-support objective forbids extra direction/pool normalization")
        if self.auxiliary.caption_loss_weight <= 0 or self.auxiliary.fsc_loss_weight <= 0:
            raise ConfigError("strong-control loss weights must be positive")
        if self.auxiliary.fsc_loss_name not in {"cross_entropy", "focal_loss"}:
            raise ConfigError("fsc_loss_name must be cross_entropy or focal_loss")
        if self.auxiliary.fsc_focal_gamma < 0:
            raise ConfigError("fsc_focal_gamma must be non-negative")
        if not 0 <= self.auxiliary.fsc_label_smoothing < 1:
            raise ConfigError("fsc_label_smoothing must be in [0,1)")
        if self.auxiliary.fsc_similarity_normalizer != "minmax":
            raise ConfigError("the initial FSC control requires minmax similarity normalization")
        if self.training.gradient_accumulation_steps != 1:
            raise ConfigError("initial contrastive protocol requires gradient_accumulation_steps=1")
        if self.training.optimizer != "upstream_bertadam":
            raise ConfigError("initial baseline must retain upstream_bertadam")
        if (
            self.training.learning_rate,
            self.training.betas,
            self.training.epsilon,
            self.training.weight_decay,
            self.training.warmup_fraction,
            self.training.schedule,
            self.training.max_grad_norm,
        ) != (0.00001, (0.9, 0.98), 0.000001, 0.001, 0.1, "upstream_warmup_cosine", 1.0):
            raise ConfigError("training optimizer fields must match the fixed UPRet defaults")
        if not self.training.restart_optimizer_for_finetuning:
            raise ConfigError("fine-tuning optimizer/schedule must restart for every arm")
        if self.training.checkpoint_every_steps < 1:
            raise ConfigError("checkpoint_every_steps must be positive")
        if self.evaluation.test_during_training:
            raise ConfigError("test_during_training must remain false")
        if not self.evaluation.complete_candidate_pool:
            raise ConfigError("evaluation must use the complete candidate pool")
        if self.evaluation.tie_policy != "stable_manifest_order":
            raise ConfigError("corrected evaluator requires stable_manifest_order")


T = TypeVar("T")


def _strict_dataclass(cls: type[T], value: Any, section: str) -> T:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ConfigError(f"unknown fields in {section}: {', '.join(unknown)}")
    missing = [
        name
        for name, item in fields.items()
        if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING and name not in value
    ]
    if missing:
        raise ConfigError(f"missing fields in {section}: {', '.join(missing)}")
    hints = get_type_hints(cls)
    converted = dict(value)
    for name, expected in hints.items():
        if name not in converted:
            continue
        raw = converted[name]
        if expected is bool and type(raw) is not bool:
            raise ConfigError(f"{section}.{name} must be a YAML boolean, not {raw!r}")
        if expected == tuple[float, float]:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ConfigError(f"{section}.{name} must contain exactly two numbers")
            converted[name] = (float(raw[0]), float(raw[1]))
    try:
        return cls(**converted)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid {section}: {error}") from error


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        missing = sorted({name for name in _ENV_PATTERN.findall(value) if name not in os.environ})
        if missing:
            raise ConfigError(f"unresolved environment variables: {', '.join(missing)}")
        expanded = _ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], value)
        if _ENV_PATTERN.search(expanded):
            raise ConfigError(f"unresolved placeholder in {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> Method1Config:
    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read config {source}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ConfigError("top-level config must be a mapping")
    raw = _expand_environment(dict(raw))
    section_types = {
        "data": DataConfig,
        "model": ModelConfig,
        "reference": ReferenceConfig,
        "miner": MinerConfig,
        "auxiliary": AuxiliaryConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "output": OutputConfig,
    }
    allowed = {
        "schema_version",
        "method_version",
        "baseline_version",
        "upstream_commit",
        "seed",
        *section_types,
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown top-level fields: {', '.join(unknown)}")
    missing = sorted(allowed - set(raw))
    if missing:
        raise ConfigError(f"missing top-level fields: {', '.join(missing)}")
    sections = {
        name: _strict_dataclass(cls, raw[name], name) for name, cls in section_types.items()
    }
    config = Method1Config(
        schema_version=raw["schema_version"],
        method_version=raw["method_version"],
        baseline_version=raw["baseline_version"],
        upstream_commit=raw["upstream_commit"],
        seed=raw["seed"],
        source_path=str(source),
        **sections,
    )
    config.validate()
    return config
