"""C19: low-rank, global cross-stream messages before native SEDS fusion.

Borrowed principle: Nagrani et al., Attention Bottlenecks for Multimodal Fusion
(NeurIPS 2021), Sec3.2.3. This two-pass adapter is NOT their full MBT architecture.
No new modality, text-conditioned gate, retrieval-score correction or auxiliary loss.
"""
from types import MethodType
import torch
from torch import nn


class BottleneckDirection(nn.Module):
    def __init__(self, width=512, rank=64, slots=4):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.seeds = nn.Parameter(torch.empty(slots, rank))
        nn.init.normal_(self.seeds, std=.02)
        self.key = nn.Linear(width, rank, bias=False)
        self.value = nn.Linear(width, rank, bias=False)
        self.query = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.scale = rank ** -.5

    def forward(self, target, source, valid):
        # Sanitize padding BEFORE normalization/projection; exclude CLS as well.
        s = self.norm(source.masked_fill(~valid[..., None], 0))
        t = self.norm(target.masked_fill(~valid[..., None], 0))
        logits = torch.einsum('kr,btr->bkt', self.seeds, self.key(s)) * self.scale
        logits = logits.masked_fill(~valid[:, None, :], torch.finfo(logits.dtype).min)
        weights = logits.softmax(-1).masked_fill(~valid[:, None, :], 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        slots = torch.bmm(weights, self.value(s))
        read = (torch.bmm(self.query(t), slots.transpose(1, 2)) * self.scale).softmax(-1)
        return self.output(torch.bmm(read, slots)).masked_fill(~valid[..., None], 0)


class GlobalExchange(nn.Module):
    def __init__(self, width=512, rank=64, slots=4, mode='cross'):
        super().__init__()
        if mode not in ('cross', 'within', 'pose_to_rgb'):
            raise ValueError(mode)
        self.mode = mode
        self.to_pose = BottleneckDirection(width, rank, slots)
        self.to_rgb = BottleneckDirection(width, rank, slots)
        if mode == 'pose_to_rgb':
            # Preserve the native pose-local stream exactly.  Keep the inactive
            # direction in the state dict so all C19 deltas share one checkpoint
            # schema, but exclude it from training/optimizer state.
            self.to_pose.requires_grad_(False)
        self.initial_identity_passed = False
        self._identity_checked = False
        self.stats = {}

    def forward(self, pose, rgb, padding):
        if pose.ndim != 3 or pose.shape != rgb.shape or padding.shape != pose.shape[:2]:
            raise ValueError('Expected aligned [B,T,D] streams and [B,T] padding mask')
        valid = ~padding.bool()
        valid = valid.clone()
        valid[:, 0] = False  # Native CLS unchanged; only actual clips exchange.
        psource, rsource = (rgb, pose) if self.mode in ('cross', 'pose_to_rgb') else (pose, rgb)
        dp = torch.zeros_like(pose) if self.mode == 'pose_to_rgb' else self.to_pose(pose, psource, valid)
        dr = self.to_rgb(rgb, rsource, valid)
        result = (pose + dp, rgb + dr)
        if not self._identity_checked:
            # Fresh-training runner asserts this; loading learned deltas for
            # inference must NOT force them back to identity or reject them.
            self.initial_identity_passed = torch.equal(result[0], pose) and torch.equal(result[1], rgb)
            self._identity_checked = True
        if self.training:
            self.stats = {
                'exchange_pose_relative_norm': float(dp.detach().float().norm() / pose.detach().float().norm().clamp_min(1e-6)),
                'exchange_rgb_relative_norm': float(dr.detach().float().norm() / rgb.detach().float().norm().clamp_min(1e-6))}
        return result


def attach_global_exchange(model, mode='cross'):
    fusion = model.fusion
    if hasattr(fusion, 'global_exchange'):
        raise ValueError('C19 already attached')
    proto = next(fusion.parameters())
    module = GlobalExchange(mode=mode).to(device=proto.device, dtype=proto.dtype)
    fusion.add_module('global_exchange', module)
    original = fusion.forward

    def forward(self, pose, rgb, mask):
        p, r = self.global_exchange(pose, rgb, mask)
        return original(p, r, mask)

    fusion.forward = MethodType(forward, fusion)
    return module


def configure_exchange_trainability(model):
    """Reapply the registered direction after generic fusion policies run.

    ``configure_trainable_stage(..., 'fusion')`` intentionally enables every
    fusion parameter. C19-R1 is asymmetric, so its inactive reverse direction
    must be frozen after that policy and before optimizer creation.
    """
    module = model.fusion.global_exchange
    module.requires_grad_(True)
    if module.mode == 'pose_to_rgb':
        module.to_pose.requires_grad_(False)
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def exchange_learning_rates(optimizer, model):
    """Split new adapter out of fusion while preserving all old group options."""
    names = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        new = [p for p in group['params'] if p.requires_grad and names[id(p)].startswith('fusion.global_exchange.')]
        old = [p for p in group['params'] if p.requires_grad and not names[id(p)].startswith('fusion.global_exchange.')]
        if old:
            groups.append(dict(group, params=old))
        if new:
            groups.append(dict(group, params=new, lr=1e-4, adaptation_group='global_exchange'))
    if not any(g['adaptation_group'] == 'global_exchange' for g in groups):
        raise ValueError('Missing trainable C19 module')
    optimizer.param_groups[:] = groups
    return [dict(adaptation_group=g['adaptation_group'],lr=g['lr'],
                 parameters=sum(p.numel() for p in g['params'])) for g in groups]
