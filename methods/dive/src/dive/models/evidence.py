from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EvidenceError(ValueError):
    """Evidence features or freeze/clone contracts are invalid."""


class LocalPoseEncoder(Protocol):
    def __call__(self, pose: Tensor, grid: Tensor) -> Tensor: ...


def _batch_norm_modules(module: nn.Module) -> tuple[nn.modules.batchnorm._BatchNorm, ...]:
    return tuple(
        child
        for child in module.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    )


class EvidenceEncoder(nn.Module):
    """Local pose/RGB fusion followed by the DIVE pointwise projector."""

    def __init__(
        self,
        pose_encoder: nn.Module,
        *,
        rgb_dim: int,
        pose_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        normalize_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(rgb_dim, pose_dim, hidden_dim, output_dim) <= 0:
            raise EvidenceError("all evidence dimensions must be positive")
        if normalize_epsilon <= 0:
            raise EvidenceError("normalize_epsilon must be positive")
        self.pose_encoder = pose_encoder
        self.rgb_dim = int(rgb_dim)
        self.pose_dim = int(pose_dim)
        self.output_dim = int(output_dim)
        self.normalize_epsilon = float(normalize_epsilon)
        self.layer_norm = nn.LayerNorm(rgb_dim + pose_dim)
        self.projector_in = nn.Linear(rgb_dim + pose_dim, hidden_dim)
        self.projector_out = nn.Linear(hidden_dim, output_dim)
        self.enforce_frozen_batch_norm()

    def enforce_frozen_batch_norm(self) -> None:
        for module in _batch_norm_modules(self):
            if not module.track_running_stats:
                raise EvidenceError("main profile requires BN running statistics")
            module.eval()
            if module.affine:
                assert module.weight is not None and module.bias is not None
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)

    def train(self, mode: bool = True) -> "EvidenceEncoder":
        super().train(mode)
        self.enforce_frozen_batch_norm()
        return self

    def forward(
        self,
        pose: Tensor,
        rgb_local: Tensor,
        grid: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if rgb_local.ndim != 3 or rgb_local.shape[-1] != self.rgb_dim:
            raise EvidenceError(f"rgb_local must have shape [B,N,{self.rgb_dim}]")
        if valid_mask.dtype != torch.bool or valid_mask.shape != rgb_local.shape[:2]:
            raise EvidenceError("valid_mask must be bool [B,N] with True=valid")
        pose_local = self.pose_encoder(pose, grid)
        expected = (*rgb_local.shape[:2], self.pose_dim)
        if tuple(pose_local.shape) != expected:
            raise EvidenceError(f"pose encoder returned {tuple(pose_local.shape)}, expected {expected}")
        if not torch.isfinite(rgb_local[valid_mask]).all() or not torch.isfinite(
            pose_local[valid_mask]
        ).all():
            raise EvidenceError("valid local features contain NaN/Inf")
        fused = torch.cat((rgb_local, pose_local), dim=-1)
        fused = torch.where(valid_mask[..., None], fused, torch.zeros_like(fused))
        projected = self.projector_out(F.gelu(self.projector_in(self.layer_norm(fused))))
        norms = torch.linalg.vector_norm(projected[valid_mask], dim=-1)
        if norms.numel() and bool((norms <= self.normalize_epsilon).any()):
            raise EvidenceError("collapsed valid evidence vector before normalization")
        normalized = F.normalize(projected, dim=-1, eps=self.normalize_epsilon)
        return torch.where(valid_mask[..., None], normalized, torch.zeros_like(normalized))


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_tensors(module: nn.Module) -> dict[str, Tensor]:
    return {name: value for name, value in module.state_dict(keep_vars=True).items()}


def assert_equal_independent_state(first: nn.Module, second: nn.Module) -> None:
    first_state = _state_tensors(first)
    second_state = _state_tensors(second)
    if first_state.keys() != second_state.keys():
        raise EvidenceError("reference/student state keys differ")
    for name in first_state:
        left = first_state[name]
        right = second_state[name]
        if not torch.equal(left.detach(), right.detach()):
            raise EvidenceError(f"reference/student state differs at {name}")
        if left.numel() and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
            raise EvidenceError(f"reference/student share storage at {name}")


@dataclass(frozen=True)
class EvidencePair:
    reference: EvidenceEncoder
    student: EvidenceEncoder
    initial_state_hash: str


def clone_reference_and_student(warmup_winner: EvidenceEncoder) -> EvidencePair:
    reference = copy.deepcopy(warmup_winner)
    student = copy.deepcopy(warmup_winner)
    reference.eval().requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.enforce_frozen_batch_norm()
    assert_equal_independent_state(reference, student)
    reference_hash = state_hash(reference)
    if reference_hash != state_hash(student):
        raise EvidenceError("reference/student hashes differ at initialization")
    return EvidencePair(reference=reference, student=student, initial_state_hash=reference_hash)
