"""C04: FP32 low-rank updates to existing upper visual attention projections.

Uses ordinary LoRA parameterization, not a novelty claim. The native scorer,
text input, attention operator and feature extraction remain unchanged.
"""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankUpdate(nn.Module):
    def __init__(self, weight, rank=8, alpha=8.):
        super().__init__()
        if weight.ndim != 2 or not 1 <= rank <= min(weight.shape):
            raise ValueError('Expected a matrix and a valid rank')
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Expected positive finite alpha')
        self.scale = alpha/rank
        self.A = nn.Parameter(torch.empty(rank,weight.shape[1],device=weight.device,dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(weight.shape[0],rank,device=weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))

    def forward(self, original):
        update = (self.B@self.A)*self.scale
        return (original.float()+update).to(original.dtype)


def attach_visual_lora(model, rank=8, alpha=8., upper_blocks=1):
    """Call once after base load, before DDP/optimizer; freeze base projections.

    Reattach the same parameterization before strict-loading a LoRA checkpoint.
    Caller selects trainable fusion and training/eval modes independently.
    """
    attached = []
    for stream in ('clip','clip_rgb'):
        blocks = getattr(model,stream).visual.transformer.resblocks
        if not 1 <= upper_blocks <= len(blocks):
            raise ValueError('Invalid number of upper blocks')
        for index in range(len(blocks)-upper_blocks,len(blocks)):
            attention = blocks[index].attn
            for module,name,label in [(attention,'in_proj_weight','qkv'),
                                      (attention.out_proj,'weight','output')]:
                if parametrize.is_parametrized(module,name):
                    raise ValueError('Projection already parameterized')
                original = getattr(module,name)
                original.requires_grad_(False)
                parametrization = LowRankUpdate(original,rank,alpha)
                parametrize.register_parametrization(module,name,parametrization)
                attached.append((f'{stream}.visual.transformer.resblocks.{index}.{label}',parametrization))
    return attached


def attach_text_lora(model, rank=8, alpha=8.):
    """Adapt shared text encoder actually called by SEDS (model.clip only)."""
    blocks = model.clip.transformer.resblocks
    attention = blocks[-1].attn
    attached = []
    for module, name, label in [(attention, 'in_proj_weight', 'qkv'),
                                (attention.out_proj, 'weight', 'output')]:
        if parametrize.is_parametrized(module, name):
            raise ValueError('Text projection already parameterized')
        original = getattr(module, name)
        original.requires_grad_(False)
        update = LowRankUpdate(original, rank, alpha)
        parametrize.register_parametrization(module, name, update)
        attached.append((f'clip.transformer.resblocks.{len(blocks)-1}.{label}', update))
    return attached
