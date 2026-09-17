"""Transparent frozen-feature scorer and bounded diagnostic readouts."""
import torch
from torch import nn
from torch.nn import functional as F


def channels(v, t, vm, tm, scale, *, masked=False):
    v, t = F.normalize(v, dim=-1), F.normalize(t, dim=-1)
    a = torch.einsum('ifd,jld->ijfl', v, t)
    vv, tv = vm == 0, tm == 1
    assert vv.any(-1).all() and tv.any(-1).all()
    logits_v = a / .07
    logits_t = a / .07
    if masked:
        logits_v = logits_v.masked_fill(~tv[None, :, None, :], -torch.inf)
        logits_t = logits_t.masked_fill(~vv[:, None, :, None], -torch.inf)
    av = (a * logits_v.softmax(-1)).sum(-1)
    at = (a * logits_t.softmax(-2)).sum(-2)
    sv = (av * vv[:, None]).sum(-1) / vv.sum(-1)[:, None]
    st = (at * tv[None]).sum(-1) / tv.sum(-1)[None]
    return scale * sv, scale * st


def balanced_loss(a, b):
    target = torch.arange(len(a), device=a.device)
    return sum(F.cross_entropy(x, target) for x in (a, a.T, b, b.T)) / 4


def pooled(x, mask):
    return (x * mask[..., None]).sum(1) / mask.sum(1)[:, None]


class Readout(nn.Module):
    """Same parameters/FLOPs for pooled control and token-interaction probe.

    The control broadcasts pooled vectors over token slots. Both execute the
    same pairwise operations and have all parameters active. A zero final layer
    makes initialization exactly R0. This is a diagnostic, not a novel method.
    """
    def __init__(self, mode='interaction', dim=64):
        super().__init__()
        self.mode = mode
        self.vproj = nn.Sequential(nn.Linear(512, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tproj = nn.Sequential(nn.Linear(512, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pair = nn.Sequential(nn.Linear(4*dim+4, 128), nn.GELU(), nn.Linear(128, 2))
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)

    def forward(self, v, t, vm, tm):
        vv, tv = vm == 0, tm == 1
        v, t = F.normalize(v, dim=-1), F.normalize(t, dim=-1)
        if self.mode == 'pooled':
            v = pooled(v, vv)[:, None].expand_as(v)
            t = pooled(t, tv)[:, None].expand_as(t)
        elif self.mode == 'zero':
            v, t = v*0, t*0
        elif self.mode != 'interaction':
            raise ValueError(self.mode)
        v, t = self.vproj(v), self.tproj(t)
        pv, pt = pooled(v, vv), pooled(t, tv)
        a = torch.einsum('ifd,jld->ijfl', F.normalize(v, dim=-1), F.normalize(t, dim=-1))
        valid = vv[:, None, :, None] & tv[None, :, None, :]
        count = valid.sum((-1, -2))
        mean = (a * valid).sum((-1, -2)) / count
        second = (a.square() * valid).sum((-1, -2)) / count
        maxv = a.masked_fill(~tv[None, :, None, :], -torch.inf).max(-1).values
        maxt = a.masked_fill(~vv[:, None, :, None], -torch.inf).max(-2).values
        maxv = (maxv * vv[:, None]).sum(-1) / vv.sum(-1)[:, None]
        maxt = (maxt * tv[None]).sum(-1) / tv.sum(-1)[None]
        left, right = torch.broadcast_tensors(pv[:, None], pt[None])
        pair = torch.cat((left, right, left*right, (left-right).abs(),
                          torch.stack((mean, second, maxv, maxt), -1)), -1)
        return self.pair(pair).unbind(-1)
