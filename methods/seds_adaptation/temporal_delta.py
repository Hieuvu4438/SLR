"""C06: masked adjacent-token differences before native fusion/scoring.

No score residual, query conditioning, pooling weights or temporal reordering.
CLS and padded tokens are preserved; only valid neighboring clips contribute.
"""
from types import MethodType
import torch
from torch import nn
from .train_policies import configure_trainable_stage, training_modes


class TemporalDelta(nn.Module):
    def __init__(self, width, rank=32):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.down = nn.Linear(2*width, rank, bias=False)
        self.output = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, x, padding_mask):
        if x.ndim != 3 or padding_mask.shape != x.shape[:2]:
            raise ValueError('Expected B,T,D tokens and B,T padding mask (1=pad)')
        valid = padding_mask == 0
        valid = valid.clone()
        valid[:, 0] = False  # never treat CLS as a neighboring clip
        z = self.norm(x.float())
        pairs = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
        difference = torch.where(pairs, z[:, 1:]-z[:, :-1], torch.zeros_like(z[:, 1:]))
        zero = torch.zeros_like(z[:, :1])
        backward = torch.cat((zero, difference), dim=1)
        forward = torch.cat((difference, zero), dim=1)
        correction = self.output(torch.nn.functional.gelu(self.down(torch.cat((backward, forward),dim=-1))))
        correction = torch.where(valid.unsqueeze(-1), correction, torch.zeros_like(correction))
        return x + correction.to(x.dtype)


def attach_temporal_delta(model, rank=32):
    if hasattr(model, 'temporal_delta'):
        raise ValueError('Temporal delta already attached')
    modules = nn.ModuleDict({name: TemporalDelta(getattr(model,stream).visual.proj.shape[1],rank)
                            for name,stream in [('pose','clip'),('rgb','clip_rgb')]})
    model.add_module('temporal_delta', modules.to(device=next(model.parameters()).device))
    original = model.get_visual_output

    def adapted(self, *args, **kwargs):
        mask, pose, rgb = original(*args, **kwargs)
        return mask, self.temporal_delta['pose'](pose,mask), self.temporal_delta['rgb'](rgb,mask)

    model.get_visual_output = MethodType(adapted, model)
    return modules


def configure_temporal_fusion(model):
    active = configure_trainable_stage(model, 'fusion')
    for parameter in model.temporal_delta.parameters():
        parameter.requires_grad_(True)
    active.append(model.temporal_delta)
    training_modes(model,active)
    return active
