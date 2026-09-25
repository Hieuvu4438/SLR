"""Numerically-equivalent speedups of upstream SEDS (no change in the computed function).

`get_sign_output` upstream builds per-clip pose windows with a Python double loop over
(batch, clip) that issues one slice-copy + one host sync per element (~2k per batch). Here
the same windows are gathered with a single advanced-indexing op. Equivalence (forward and
gradients) is asserted in tests/test_fast_seds.py.
"""
import torch

from modules.modeling import CLIP4Clip


def gather_clip_windows(feat, clips_start, clip_mask, slide_windows):
    """feat: [B, T, D]; clips_start: [B, L] (-1 = padding); clip_mask: [B, L+1] (1 = padding,
    index 0 is the CLS slot). Returns [B, L, W, D] with zeros at padded clips."""
    B, L = clips_start.shape
    valid = clips_start != -1
    # Upstream asserts the same mask/clip-start consistency inside its loop.
    assert torch.equal(valid, clip_mask[:, 1:] == 0)
    offs = torch.arange(slide_windows, device=feat.device)
    idx = clips_start.clamp(min=0)[:, :, None] + offs[None, None, :]  # [B, L, W]
    assert int(idx.max()) < feat.shape[1]
    b_idx = torch.arange(B, device=feat.device)[:, None, None]
    out = feat[b_idx, idx]  # [B, L, W, D]
    return out * valid[:, :, None, None].to(out.dtype)


class FastCLIP4Clip(CLIP4Clip):
    def get_sign_output(self, right_batch, left_batch, body_batch):
        clips_start = body_batch['clips_start']
        clip_mask = body_batch['mask']
        rgb_feature = body_batch['rgb']
        batch_num, feature_len = clips_start.size()
        slide_windows = self.task_config.slide_windows
        pose_all = {'right': right_batch['pose'], 'left': left_batch['pose'], 'body': body_batch['pose']}
        pose_all = self.signbert.gcn_emb(pose_all)
        feat = pose_all['feat']
        feat_dim = feat.shape[-1]
        pose_final_new = gather_clip_windows(feat, clips_start, clip_mask, slide_windows)
        pose_final_new = pose_final_new.reshape(batch_num * feature_len, slide_windows, feat_dim)
        pose_final_new = self.signbert.sign_conv(pose_final_new)
        pose_final_new = pose_final_new.reshape(batch_num, feature_len, slide_windows, feat_dim)
        pose_final_new = torch.mean(pose_final_new, dim=-2)
        pose_final_new = pose_final_new.permute(0, 2, 1).unsqueeze(-1)
        return rgb_feature, pose_final_new, clip_mask
