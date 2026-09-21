"""Loss-only adaptation: retain the native scorer and all inference paths."""
import math


def fused_priority_loss(native_losses, auxiliary_weight=0.25):
    """Scale pose/text and RGB/text auxiliaries; leave cross-stream matching intact.

    Native tuple: total, fusion, pose, rgb, pose_kl, rgb_kl, rgb_pose_match.
    Weight 1 is the existing (non-freeze_exfusion) training objective.
    """
    if len(native_losses) != 7:
        raise ValueError("Expected the seven native SEDS loss components")
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("Auxiliary weight must be finite and nonnegative")
    _, fusion, pose, rgb, pose_kl, rgb_kl, match = native_losses
    return fusion + auxiliary_weight * pose + auxiliary_weight * rgb + pose_kl + rgb_kl + match
