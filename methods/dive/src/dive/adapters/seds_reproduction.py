from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PINNED_SEDS_COMMIT = "434e3f714fcb6a7d1f4001fb9a246bbd93ec0246"
_SCHEMA_VERSION = "seds_how2sign_reproduction.v1"

_SOURCE_SCRIPTS = {
    "eval": {
        "path": "scripts/eval_h2s.sh",
        "sha256": "3692b5a72746d16aa62788073e2b1de81c19c8736d7b06dcaadec7077b68ce29",
    },
    "train": {
        "path": "scripts/train_h2s.sh",
        "sha256": "c76563b8f334d3af88f76f0b800500a42689bd5027191caec9bd1032911cef58",
    },
}

_PUBLISHED_EVAL_ARGUMENTS: dict[str, Any] = {
    "batch_size_val": 64,
    "coef_lr": 1.0,
    "crop_size": 256,
    "data_path": "data_h2",
    "datatype": "h2s_pose",
    "do_eval": True,
    "feature_len": 64,
    "features_RGB_path": "./How2Sign/I3D_features/",
    "features_path": "./How2Sign/RTMpose/Pose_all_24rates/",
    "frames_threshold": 0.1,
    "freeze_layer_num": 0,
    "fusion_type": "gloss_atten",
    "init_model": "ckpts/h2s_best_model.bin",
    "linear_patch": "2d",
    "lr": 1e-5,
    "max_length_frames": 300,
    "max_words": 32,
    "original_size_h": 256,
    "original_size_w": 256,
    "output_dir": "result_eval/eval_h2s",
    "pretrained_clip_name": "ViT-B/32",
    "rgb_pose_match": True,
    "rgb_pose_match_loss": 0.4,
    "sign_lr": 1e-4,
    "signbert": True,
    "sim_header": "Filip",
    "slide_windows": 16,
    "threshold": 0.4,
    "windows_stride": 1,
}

_PUBLISHED_TRAIN_ARGUMENTS: dict[str, Any] = {
    "batch_size": 128,
    "batch_size_val": 64,
    "coef_lr": 1.0,
    "crop_size": 256,
    "data_path": "data_h2",
    "datatype": "h2s_pose",
    "do_train": True,
    "epochs": 200,
    "feature_len": 64,
    "features_RGB_path": "./How2Sign/I3D_features/",
    "features_path": "./How2Sign/RTMpose/Pose_all_24rates/",
    "frames_threshold": 0.1,
    "freeze_layer_num": 0,
    "fusion_type": "gloss_atten",
    "init_sign_model": "ckpt/pretrain_signbert.pth",
    "linear_patch": "2d",
    "lr": 1e-5,
    "max_length_frames": 300,
    "max_words": 32,
    "n_display": 10,
    "num_thread_reader": 64,
    "original_size_h": 256,
    "original_size_w": 256,
    "output_dir": "result_train/h2s",
    "pretrained_clip_name": "ViT-B/32",
    "rgb_pose_match": True,
    "rgb_pose_match_loss": 0.4,
    "sign_lr": 1e-4,
    "signbert": True,
    "sim_header": "Filip",
    "slide_windows": 16,
    "threshold": 0.4,
    "windows_stride": 1,
}

_MODEL_ARGUMENTS: dict[str, Any] = {
    "aug_choose": "t2v",
    "cross_model": "cross-base",
    "dropout": 0.1,
    "feature_len": 64,
    "freeze_exfusion": False,
    "fusion_type": "gloss_atten",
    "hidden_dim": 512,
    "in_channels": 2,
    "kl_logit": 0.01,
    "kl_pose_loss": 0.5,
    "kl_rgb_loss": 0.5,
    "layout_encoder": "stb",
    "linear_patch": "2d",
    "mix_design": "balance",
    "not_load_visual": False,
    "pose_dim": 1536,
    "pretrained_clip_name": "ViT-B/32",
    "rgb_dim": 1024,
    "rgb_pose_kl": False,
    "rgb_pose_match": True,
    "rgb_pose_match_loss": 0.4,
    "signbert": True,
    "sim_header": "Filip",
    "slide_windows": 16,
    "strategy": "spatial",
    "temporal_pad": 0,
    "visual_num_hidden_layers": 12,
    "windows_stride": 1,
}

_EXTERNAL_ASSETS = {
    "clip_initialization": "modules/ViT-B-32.pt",
    "locked_checkpoint": "ckpts/h2s_best_model.bin",
    "pose_features": "How2Sign/RTMpose/Pose_all_24rates",
    "rgb_features": "How2Sign/I3D_features",
    "signbert_initialization": "ckpt/pretrain_signbert.pth",
    "tokenizer": "modules/bpe_simple_vocab_16e6.txt.gz",
}

_CONTROLLED_PROTOCOL = {
    "checkpoint_selection": "independent_dev_only",
    "dev_source": "local_labels.dev.json",
    "name": "seds_how2sign_controlled_v1",
    "test_access": "single_locked_final_evaluation",
    "test_source": "pinned_data_h2/test.pkl",
    "train_source": "pinned_data_h2/train.pkl",
}

_CONTROLLED_TRAINING: dict[str, Any] = {
    "augmentation": "vendored_textaugment_eda_random_swap_n1_probability_0.5",
    "beta1": 0.9,
    "beta2": 0.98,
    "checkpoint_retention": "rolling_resume_plus_earliest_best",
    "effective_batch_size": 128,
    "epochs": 200,
    "epsilon": 1e-6,
    "global_grad_clip_norm": 1.0,
    "gradient_accumulation_steps": 1,
    "lr_clip": 1e-5,
    "lr_other": 1e-4,
    "lr_signbert": 1e-4,
    "objective": "published_seds_fusion_pose_rgb_and_pose_rgb_match",
    "optimizer": "pinned_bertadam",
    "optimizer_per_parameter_clip_norm": 1.0,
    "precision": "float32",
    "sampling_unit": "unique_text_id_uniform_video_view_each_epoch",
    "schedule": "warmup_cosine",
    "selection_metric": "mean_t2v_v2t_r1",
    "selection_split": "dev",
    "tie_break": "earliest_optimizer_step_then_epoch",
    "warmup_fraction": 0.1,
    "weight_decay": 0.001,
    "world_size": 1,
}


class SedsReproductionError(ValueError):
    """The checked native SEDS reproduction artifact is missing or inconsistent."""


@dataclass(frozen=True)
class SedsReproduction:
    model_arguments: Mapping[str, Any]
    published_eval_arguments: Mapping[str, Any]
    published_train_arguments: Mapping[str, Any]
    external_assets: Mapping[str, str]
    controlled_protocol: Mapping[str, str]
    controlled_training: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SedsReproductionError(f"{name} must be a mapping")
    return dict(value)


def _assert_typed_exact(name: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        raise SedsReproductionError(f"{name} keys mismatch; missing={missing}, unknown={unknown}")
    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            mismatches[key] = (actual_value, expected_value)
    if mismatches:
        raise SedsReproductionError(f"{name} values/types mismatch: {mismatches}")


def load_seds_reproduction(
    path: str | Path, *, upstream_root: str | Path | None = None
) -> SedsReproduction:
    """Load the one typed How2Sign contract and optionally verify its pinned source scripts."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SedsReproductionError("SEDS reproduction config is unreadable") from exc
    document = _mapping(raw, "SEDS reproduction config")
    expected_top = {
        "schema_version",
        "upstream_commit",
        "source_scripts",
        "published_eval",
        "published_train",
        "model_arguments",
        "external_assets",
        "controlled_protocol",
        "controlled_training",
    }
    if set(document) != expected_top:
        raise SedsReproductionError("SEDS reproduction config has missing or unknown sections")
    if document["schema_version"] != _SCHEMA_VERSION:
        raise SedsReproductionError(f"schema_version must be {_SCHEMA_VERSION}")
    if document["upstream_commit"] != PINNED_SEDS_COMMIT:
        raise SedsReproductionError("SEDS reproduction config has the wrong upstream commit")

    scripts = _mapping(document["source_scripts"], "source_scripts")
    _assert_typed_exact("source_scripts", scripts, _SOURCE_SCRIPTS)
    published_eval = _mapping(document["published_eval"], "published_eval")
    published_train = _mapping(document["published_train"], "published_train")
    model_arguments = _mapping(document["model_arguments"], "model_arguments")
    external_assets = _mapping(document["external_assets"], "external_assets")
    controlled_protocol = _mapping(document["controlled_protocol"], "controlled_protocol")
    controlled_training = _mapping(document["controlled_training"], "controlled_training")
    _assert_typed_exact("published_eval", published_eval, _PUBLISHED_EVAL_ARGUMENTS)
    _assert_typed_exact("published_train", published_train, _PUBLISHED_TRAIN_ARGUMENTS)
    _assert_typed_exact("model_arguments", model_arguments, _MODEL_ARGUMENTS)
    _assert_typed_exact("external_assets", external_assets, _EXTERNAL_ASSETS)
    _assert_typed_exact("controlled_protocol", controlled_protocol, _CONTROLLED_PROTOCOL)
    _assert_typed_exact("controlled_training", controlled_training, _CONTROLLED_TRAINING)

    if upstream_root is not None:
        root = Path(upstream_root).resolve()
        for label, record in _SOURCE_SCRIPTS.items():
            script = root / record["path"]
            if not script.is_file() or _sha256(script) != record["sha256"]:
                raise SedsReproductionError(
                    f"pinned SEDS {label} source script is missing or has changed: {script}"
                )

    return SedsReproduction(
        model_arguments=model_arguments,
        published_eval_arguments=published_eval,
        published_train_arguments=published_train,
        external_assets=external_assets,
        controlled_protocol=controlled_protocol,
        controlled_training=controlled_training,
    )


def verify_seds_checkout(upstream_root: str | Path) -> Path:
    """Require the exact clean detached SEDS source used by every native adapter path."""
    root = Path(upstream_root).resolve()
    if not (root / ".git").exists():
        raise SedsReproductionError(f"SEDS checkout is not a Git worktree: {root}")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise SedsReproductionError(f"cannot inspect SEDS checkout: {root}") from exc
    if head != PINNED_SEDS_COMMIT:
        raise SedsReproductionError("SEDS checkout commit differs from the pinned revision")
    if dirty:
        raise SedsReproductionError("SEDS checkout must be clean for controlled provenance")
    return root
