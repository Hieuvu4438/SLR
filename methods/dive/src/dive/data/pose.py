from __future__ import annotations

import torch
from torch import Tensor


class PoseError(ValueError):
    """Pose data or its locality-preserving preprocessing is invalid."""


def normalize_pose_per_step(
    pose: Tensor,
    joint_valid: Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Center/scale each input step independently and retain a bool validity mask.

    This deliberately does not interpolate across time, so preprocessing adds no temporal RF.
    Coordinates after the first two channels (for example confidence) are copied for valid joints.
    """
    if pose.ndim != 3 or pose.shape[-1] < 2:
        raise PoseError("pose must have shape [steps,joints,channels>=2]")
    if joint_valid.dtype != torch.bool or joint_valid.shape != pose.shape[:2]:
        raise PoseError("joint_valid must be bool [steps,joints] with True=valid")
    if epsilon <= 0:
        raise PoseError("epsilon must be positive")
    if not torch.isfinite(pose[joint_valid]).all():
        raise PoseError("valid pose coordinates must be finite")
    count = joint_valid.sum(dim=1, keepdim=True)
    step_valid = count.squeeze(1) >= 2
    safe_count = count.clamp_min(1).to(pose.dtype)
    xy = pose[..., :2]
    masked_xy = torch.where(joint_valid[..., None], xy, torch.zeros_like(xy))
    center = masked_xy.sum(dim=1, keepdim=True) / safe_count[..., None]
    centered = xy - center
    squared_radius = torch.where(
        joint_valid,
        centered.square().sum(dim=-1),
        torch.zeros_like(centered[..., 0]),
    )
    scale = torch.sqrt(squared_radius.sum(dim=1, keepdim=True) / safe_count).clamp_min(epsilon)
    normalized_xy = centered / scale[..., None]
    output = pose.clone()
    output[..., :2] = normalized_xy
    output = torch.where(joint_valid[..., None] & step_valid[:, None, None], output, 0.0)
    return output, joint_valid & step_valid[:, None]
