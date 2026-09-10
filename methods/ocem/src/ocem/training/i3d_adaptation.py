"""CiCo-compatible preprocessing and one-step P14T I3D adaptation validation."""

from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from ocem.data.pseudoclips import decode_pseudoclip, load_pseudoclip_index
from ocem.provenance.hashes import canonical_json_sha256, sha256_file


class I3DAdaptationError(ValueError):
    """Raised when the pinned I3D adaptation contract cannot be reproduced."""


@dataclass(frozen=True)
class CicoAugmentationDraws:
    """All stochastic choices consumed by CiCo's batch-level GPU collater."""

    horizontal_flip: bool
    color_jitter: tuple[float, float, float]
    scale_uniform: np.ndarray
    offset_uniform: np.ndarray


def sample_cico_augmentation_draws(
    batch_size: int,
    *,
    seed: int,
    hflip_probability: float = 0.5,
    jitter: float = 0.2,
) -> CicoAugmentationDraws:
    """Replay the Python/NumPy RNG call order in CiCo's training collater."""

    if batch_size < 1 or seed < 0:
        raise I3DAdaptationError("batch_size must be positive and seed nonnegative")
    if not 0 <= hflip_probability <= 1 or jitter < 0:
        raise I3DAdaptationError("invalid augmentation probability or jitter")
    python_rng = random.Random(seed)
    numpy_rng = np.random.RandomState(seed)
    return CicoAugmentationDraws(
        horizontal_flip=python_rng.random() < hflip_probability,
        color_jitter=tuple(python_rng.uniform(1 - jitter, 1 + jitter) for _ in range(3)),
        scale_uniform=numpy_rng.rand(batch_size, 1),
        offset_uniform=numpy_rng.rand(batch_size, 2),
    )


def _bbox_format(bbox: np.ndarray, source: str, destination: str) -> np.ndarray:
    if source == "yxyx" and destination == "cenhw":
        height_width = bbox[:, 2:] - bbox[:, :2]
        center = bbox[:, :2] + height_width / 2
        return np.hstack((center, height_width))
    if source == "cenhw" and destination == "yxyx":
        minimum = bbox[:, :2] - bbox[:, 2:] / 2
        maximum = bbox[:, :2] + bbox[:, 2:] / 2
        return np.hstack((minimum, maximum))
    raise I3DAdaptationError(f"unsupported bbox conversion: {source} -> {destination}")


def _scale_bbox(bbox: np.ndarray, scale: np.ndarray) -> np.ndarray:
    center = _bbox_format(bbox, "yxyx", "cenhw")
    center[:, 2:] *= scale
    return _bbox_format(center, "cenhw", "yxyx")


def apply_cico_train_augmentation(
    rgb,
    draws: CicoAugmentationDraws,
    *,
    input_resolution: int = 224,
    resize_resolution: int = 256,
    scale_factor: float = 0.1,
):
    """Apply CiCo's batch-shared flip/jitter and per-item random crop exactly."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise I3DAdaptationError("PyTorch is required for I3D augmentation") from error
    if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (3, 16):
        raise I3DAdaptationError(f"expected [B,3,16,H,W] input, got {tuple(rgb.shape)}")
    batch_size = rgb.shape[0]
    if draws.scale_uniform.shape != (batch_size, 1) or draws.offset_uniform.shape != (
        batch_size,
        2,
    ):
        raise I3DAdaptationError("augmentation draw shapes do not match batch size")
    if resize_resolution < input_resolution or not 0 <= scale_factor < 1:
        raise I3DAdaptationError("invalid crop resolutions or scale factor")
    output = rgb.clone()
    if draws.horizontal_flip:
        output = torch.flip(output, dims=[-1])
    for channel, multiplier in enumerate(draws.color_jitter):
        output[:, channel].mul_(multiplier).clamp_(0, 1)

    boxes = np.tile(np.array([[0, 0, 1, 1]], dtype=np.float32), (batch_size, 1))
    # Preserve the upstream NumPy order: RNG draws and scale arithmetic stay
    # float64, while assignment inside scale_yxyx_bbox casts into float32 boxes.
    random_scale = 1 / (1 - scale_factor + 2 * scale_factor * draws.scale_uniform)
    boxes = _scale_bbox(boxes, random_scale)
    crop_scale = (input_resolution / resize_resolution) * random_scale
    boxes = _scale_bbox(boxes, crop_scale)
    center = _bbox_format(boxes, "yxyx", "cenhw")
    valid_region = ((1 - crop_scale) / crop_scale) * center[:, 2:]
    offsets = (draws.offset_uniform - 0.5) * valid_region
    boxes += np.tile(offsets, (1, 2))
    boxes = 2 * boxes - 1

    grids = torch.zeros(
        batch_size,
        input_resolution,
        input_resolution,
        2,
        device=output.device,
        dtype=output.dtype,
    )
    for index, box in enumerate(boxes):
        y_ticks = torch.linspace(
            float(box[0]),
            float(box[2]),
            steps=input_resolution,
            device=output.device,
            dtype=output.dtype,
        )
        x_ticks = torch.linspace(
            float(box[1]),
            float(box[3]),
            steps=input_resolution,
            device=output.device,
            dtype=output.dtype,
        )
        grid_y, grid_x = torch.meshgrid(y_ticks, x_ticks, indexing="ij")
        grids[index] = torch.stack((grid_x, grid_y), dim=2)
    output = functional.grid_sample(
        output.reshape(batch_size, 3 * 16, output.shape[-2], output.shape[-1]),
        grid=grids,
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
    ).reshape(batch_size, 3, 16, input_resolution, input_resolution)
    output[:, 0].sub_(0.5)
    output[:, 1].sub_(0.5)
    output[:, 2].sub_(0.5)
    return output


def _import_source(path: Path, module_name: str) -> ModuleType:
    if not path.is_file():
        raise I3DAdaptationError(f"pinned source is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise I3DAdaptationError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adaptation_i3d(
    *,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    implementation: str | Path,
    expected_implementation_sha256: str,
    device: str,
):
    """Strictly construct the pinned 5383-class I3D with trainable parameters."""

    try:
        import torch
    except ImportError as error:
        raise I3DAdaptationError("PyTorch is required for I3D adaptation") from error
    checkpoint, implementation = Path(checkpoint), Path(implementation)
    if sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise I3DAdaptationError("Oxford I3D checkpoint SHA-256 mismatch")
    if sha256_file(implementation) != expected_implementation_sha256:
        raise I3DAdaptationError("pinned I3D implementation SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise I3DAdaptationError("Oxford I3D checkpoint lacks state_dict")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload["state_dict"].items()
    }
    classifier = state.get("logits.conv3d.weight")
    if classifier is None or tuple(classifier.shape) != (5383, 1024, 1, 1, 1):
        raise I3DAdaptationError("Oxford I3D classifier shape mismatch")
    module = _import_source(implementation, "ocem_pinned_i3d_adaptation")
    model = module.InceptionI3d(
        num_classes=5383,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=16,
        include_embds=True,
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(True)
    try:
        return model.to(torch.device(device))
    except (RuntimeError, ValueError) as error:
        raise I3DAdaptationError(f"cannot move I3D to {device}: {error}") from error


def build_cico_sgd(
    model,
    *,
    learning_rate: float = 0.01,
    encoder_coefficient: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
):
    """Reproduce CiCo's encoder/classifier SGD parameter groups."""

    try:
        import torch
    except ImportError as error:
        raise I3DAdaptationError("PyTorch is required for I3D adaptation") from error
    if learning_rate <= 0 or encoder_coefficient <= 0 or momentum < 0 or weight_decay < 0:
        raise I3DAdaptationError("invalid SGD hyperparameters")
    core = model.module if hasattr(model, "module") else model
    named = list(core.named_parameters())
    encoder = [parameter for name, parameter in named if "logits" not in name]
    classifier = [parameter for name, parameter in named if "logits" in name]
    if not encoder or not classifier:
        raise I3DAdaptationError("could not partition I3D encoder and classifier parameters")
    return torch.optim.SGD(
        [
            {
                "params": encoder,
                "weight_decay": weight_decay,
                "lr": encoder_coefficient * learning_rate,
                "momentum": momentum,
            },
            {
                "params": classifier,
                "weight_decay": weight_decay,
                "lr": learning_rate,
                "momentum": momentum,
            },
        ]
    )


def adaptation_update(model, optimizer, inputs, targets) -> tuple[Any, dict[str, Any]]:
    """Execute the upstream train-epoch operation order for one batch."""

    try:
        import torch.nn.functional as functional
    except ImportError as error:
        raise I3DAdaptationError("PyTorch is required for I3D adaptation") from error
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = functional.cross_entropy(outputs["logits"], targets, reduction="mean")
    loss.backward()
    core = model.module if hasattr(model, "module") else model
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in core.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    return loss.detach(), gradients


def _upstream_augmentation_reference(rgb, trainer_root: Path, seed: int):
    """Execute the pinned collater math using its transform helpers as an oracle."""

    try:
        import torch
    except ImportError as error:
        raise I3DAdaptationError("PyTorch is required for augmentation parity") from error
    transforms_path = trainer_root / "utils" / "transforms.py"
    imutils_path = trainer_root / "utils" / "imutils.py"
    if not transforms_path.is_file() or not imutils_path.is_file():
        raise I3DAdaptationError("pinned trainer transform sources are missing")
    package_name = "ocem_pinned_cico_trainer_utils"
    package = ModuleType(package_name)
    package.__path__ = [str((trainer_root / "utils").resolve())]
    sys.modules[package_name] = package
    transforms = _import_source(transforms_path, f"{package_name}.transforms")

    random.seed(seed)
    np.random.seed(seed)
    output = rgb.clone()
    horizontal_flip = random.random() < 0.5
    if horizontal_flip:
        output = torch.flip(output, dims=[-1])
    output = transforms.im_color_jitter(output, num_in_frames=16, thr=0.2)
    batch_size = output.shape[0]
    boxes = np.zeros((batch_size, 4), dtype=np.float32)
    boxes[:] = np.array([0, 0, 1, 1])
    random_scale = np.random.rand(batch_size, 1)
    random_scale = 1 - 0.1 + 2 * 0.1 * random_scale
    random_scale = 1 / random_scale
    boxes = transforms.scale_yxyx_bbox(boxes, scale=random_scale)
    crop_scale = (224 / 256) * random_scale
    boxes = transforms.scale_yxyx_bbox(boxes, scale=crop_scale)
    center = transforms.bbox_format(boxes, src="yxyx", dest="cenhw")
    valid_region = ((1 - crop_scale) / crop_scale) * center[:, 2:]
    offsets = (np.random.rand(batch_size, 2) - 0.5) * valid_region
    boxes += np.tile(offsets, (1, 2))
    boxes = 2 * boxes - 1
    grids = torch.zeros(batch_size, 224, 224, 2, device=output.device, dtype=output.dtype)
    for index, box in enumerate(boxes):
        y_ticks = torch.linspace(float(box[0]), float(box[2]), steps=224, device=output.device)
        x_ticks = torch.linspace(float(box[1]), float(box[3]), steps=224, device=output.device)
        grid_y, grid_x = torch.meshgrid(y_ticks, x_ticks, indexing="ij")
        grids[index] = torch.stack((grid_x, grid_y), dim=2)
    output = torch.nn.functional.grid_sample(
        output.reshape(batch_size, 48, output.shape[-2], output.shape[-1]),
        grid=grids,
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
    ).reshape(batch_size, 3, 16, 224, 224)
    output = transforms.color_normalize(
        output, torch.tensor([0.5, 0.5, 0.5]), torch.tensor([1.0, 1.0, 1.0])
    )
    return output


def _seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _named_core(model) -> dict[str, Any]:
    core = model.module if hasattr(model, "module") else model
    return dict(core.named_parameters())


def _state_core(model) -> dict[str, Any]:
    core = model.module if hasattr(model, "module") else model
    return core.state_dict()


def validate_one_batch_adaptation_parity(
    *,
    index: str | Path,
    expected_index_sha256: str,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    implementation: str | Path,
    expected_implementation_sha256: str,
    trainer_root: str | Path,
    expected_transforms_sha256: str,
    device: str = "cuda:0",
    batch_size: int = 4,
    seed: int = 0,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare port and pinned-oracle augmentation, loss, gradients, and SGD update."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise I3DAdaptationError("PyTorch is required for adaptation parity") from error
    if batch_size < 1 or seed < 0 or absolute_tolerance <= 0 or relative_tolerance < 0:
        raise I3DAdaptationError("invalid parity batch size, seed, or tolerance")
    trainer_root = Path(trainer_root)
    transforms_path = trainer_root / "utils" / "transforms.py"
    if sha256_file(transforms_path) != expected_transforms_sha256:
        raise I3DAdaptationError("pinned trainer transforms SHA-256 mismatch")
    records = load_pseudoclip_index(
        index, expected_sha256=expected_index_sha256, adaptation_split="train"
    )
    if len(records) < batch_size:
        raise I3DAdaptationError("pseudo-label train split is smaller than parity batch")
    decoded = [
        decode_pseudoclip(record, training=True, rng=random.Random(seed + position))
        for position, record in enumerate(records[:batch_size])
    ]
    raw_batch = torch.stack([item["rgb"] for item in decoded])
    targets = torch.tensor([item["class"] for item in decoded], dtype=torch.long)
    draws = sample_cico_augmentation_draws(batch_size, seed=seed)
    resolved_device = torch.device(device)
    port_batch = apply_cico_train_augmentation(raw_batch.to(resolved_device), draws)
    reference_batch = _upstream_augmentation_reference(
        raw_batch.to(resolved_device), trainer_root, seed
    )
    augmentation_error = float((port_batch - reference_batch).abs().max())

    port_model = load_adaptation_i3d(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        implementation=implementation,
        expected_implementation_sha256=expected_implementation_sha256,
        device=device,
    )
    reference_core = load_adaptation_i3d(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        implementation=implementation,
        expected_implementation_sha256=expected_implementation_sha256,
        device=device,
    )
    if resolved_device.type == "cuda":
        reference_model = torch.nn.DataParallel(
            reference_core, device_ids=[resolved_device.index or 0]
        )
    else:
        reference_model = reference_core
    port_optimizer = build_cico_sgd(port_model)
    reference_optimizer = build_cico_sgd(reference_model)
    targets = targets.to(resolved_device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    deterministic_parameter_backward = resolved_device.type == "cpu"
    if deterministic_parameter_backward:
        torch.use_deterministic_algorithms(True)

    _seed_torch(seed)
    port_loss, port_gradients = adaptation_update(port_model, port_optimizer, port_batch, targets)
    _seed_torch(seed)
    reference_model.train()
    reference_optimizer.zero_grad()
    reference_outputs = reference_model(reference_batch)
    reference_loss = functional.cross_entropy(
        reference_outputs["logits"], targets, reduction="mean"
    )
    reference_loss.backward()
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in _named_core(reference_model).items()
        if parameter.grad is not None
    }
    reference_optimizer.step()

    if set(port_gradients) != set(reference_gradients):
        raise I3DAdaptationError("port/reference gradient parameter sets differ")
    gradient_errors = {
        name: float((port_gradients[name] - reference_gradients[name]).abs().max())
        for name in port_gradients
    }
    port_state, reference_state = _state_core(port_model), _state_core(reference_model)
    if set(port_state) != set(reference_state):
        raise I3DAdaptationError("port/reference post-update state sets differ")
    update_errors = {
        name: float((port_state[name] - reference_state[name]).abs().max())
        for name in port_state
        if port_state[name].is_floating_point()
    }
    gradient_max = max(gradient_errors.values(), default=0.0)
    update_max = max(update_errors.values(), default=0.0)
    gradient_worst_name = max(gradient_errors, key=gradient_errors.get)
    update_worst_name = max(update_errors, key=update_errors.get)
    gradient_allclose = all(
        torch.allclose(
            port_gradients[name],
            reference_gradients[name],
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        for name in port_gradients
    )
    update_allclose = all(
        torch.allclose(
            port_state[name],
            reference_state[name],
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        for name in port_state
    )
    loss_error = abs(float(port_loss) - float(reference_loss.detach()))
    finite = all(bool(torch.isfinite(value).all()) for value in port_gradients.values()) and all(
        bool(torch.isfinite(value).all()) for value in port_state.values()
    )
    representative_names = [
        next(name for name in port_gradients if "logits" not in name),
        next(name for name in port_gradients if "logits" in name),
    ]
    passed = (
        finite
        and augmentation_error <= absolute_tolerance
        and loss_error <= absolute_tolerance
        and gradient_allclose
        and update_allclose
    )
    return {
        "schema_version": "ocem.p14t_adaptation_step_parity.v1",
        "status": "PASS" if passed else "FAIL_TECHNICAL",
        "inputs": {
            "index": str(Path(index).resolve()),
            "index_sha256": expected_index_sha256,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": expected_checkpoint_sha256,
            "implementation": str(Path(implementation).resolve()),
            "implementation_sha256": expected_implementation_sha256,
            "trainer_transforms": str(transforms_path.resolve()),
            "trainer_transforms_sha256": expected_transforms_sha256,
            "pseudo_ids": [item["pseudo_id"] for item in decoded],
            "batch_shape": list(raw_batch.shape),
            "targets": [int(item) for item in targets.cpu()],
        },
        "recipe": {
            "seed": seed,
            "batch_size": batch_size,
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "encoder_lr_coefficient": 1.0,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "loss": "cross_entropy_mean",
            "model_mode": "train",
            "dropout_probability": 0.5,
            "reference_wrapper": (
                "torch.nn.DataParallel" if resolved_device.type == "cuda" else "none"
            ),
            "deterministic_parameter_backward": deterministic_parameter_backward,
        },
        "parity": {
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "augmentation_max_absolute_error": augmentation_error,
            "loss_port": float(port_loss),
            "loss_reference": float(reference_loss.detach()),
            "loss_absolute_error": loss_error,
            "gradient_max_absolute_error": gradient_max,
            "gradient_worst_absolute_parameter": gradient_worst_name,
            "gradient_allclose": gradient_allclose,
            "update_state_max_absolute_error": update_max,
            "update_worst_absolute_state": update_worst_name,
            "update_allclose": update_allclose,
            "gradient_parameter_count": len(port_gradients),
            "post_update_float_state_count": len(update_errors),
            "all_gradients_and_states_finite": finite,
            "representative_gradient_norms": {
                name: float(port_gradients[name].norm()) for name in representative_names
            },
            "gradient_error_digest": canonical_json_sha256(gradient_errors),
            "update_error_digest": canonical_json_sha256(update_errors),
        },
        "data_policy": {
            "adaptation_split": "train",
            "holdout_used_for_optimizer": False,
            "validation_or_test_used": False,
            "codec_roundtrip": False,
        },
        "ready_for_adaptation_training": passed,
    }
