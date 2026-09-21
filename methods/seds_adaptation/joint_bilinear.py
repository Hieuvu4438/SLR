"""C15: low-rank within-part joint bilinear statistics before anatomical pooling.

Native max pooling remains intact. No temporal weighting, cross-articulator
product, adjacency change, or raw bone descriptor is introduced.
"""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F


class JointBilinear(nn.Module):
    def __init__(self, width=256, rank=16, centered=False):
        super().__init__()
        self.width = width
        self.centered = centered
        self.input = nn.Linear(width, rank, bias=False)
        self.register_buffer('triangle', torch.triu_indices(rank, rank))
        self.output = nn.Linear(rank * (rank + 1) // 2, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def descriptor(self, x, valid):
        # x: flattened frames, channels, 1, joints; validity from native XY.
        if x.ndim != 4 or x.shape[1:3] != (self.width, 1):
            raise ValueError('Expected framewise pre-pooling GCN features')
        if valid.shape != (x.shape[0], x.shape[-1]):
            raise ValueError('Joint validity/layout mismatch')
        z = x.squeeze(2).transpose(1, 2)
        z = self.input(F.layer_norm(z, (self.width,)))
        moment = self.second_moment(z, valid)
        packed = moment[:, self.triangle[0], self.triangle[1]]
        # Smooth signed-square-root has finite derivative at zero (including
        # completely missing parts), unlike sign(x)*sqrt(abs(x)).
        packed = packed / torch.sqrt(packed.abs() + 1e-4)
        return F.normalize(packed, dim=-1, eps=1e-6)

    def second_moment(self, z, valid):
        z = z.masked_fill(~valid[..., None], 0.)
        count = valid.sum(-1).clamp_min(1)[:, None, None]
        if self.centered:
            mean = z.sum(1, keepdim=True) / count
            z = (z - mean).masked_fill(~valid[..., None], 0.)
        # Population covariance for R1; original uncentered second moment otherwise.
        # Missing/singleton parts give exactly zero covariance, without fake samples.
        return z.transpose(1, 2).bmm(z) / count

    def forward(self, x, valid):
        return self.output(self.descriptor(x, valid))[:, :, None, None]


def attach_encoder_bilinear(encoder, centered=False):
    if hasattr(encoder, 'joint_bilinear'):
        raise ValueError('Joint bilinear module already attached')
    if encoder.graph.seqlen != 1 or encoder.fc_out != 256:
        raise ValueError('Expected native framewise 256-channel anatomical pooling')
    prototype = next(encoder.parameters())
    module = JointBilinear(centered=centered).to(prototype.device, prototype.dtype)
    encoder.add_module('joint_bilinear', module)
    original_forward, original_pool = encoder.forward, encoder.graph_max_pool
    valid, part_index = None, 0

    def forward(self, xy):
        nonlocal valid, part_index
        if valid is not None:
            raise RuntimeError('Reentrant encoder is unsupported')
        if not torch.isfinite(xy).all():
            raise ValueError('Nonfinite native pose')
        valid = xy.ne(0).any(-1).reshape(-1, xy.shape[-2])
        part_index = 0
        try:
            result = original_forward(xy)
            if part_index != len(self.graph.part):
                raise ValueError('Native anatomical pooling call path changed')
            return result
        finally:
            valid = None

    def pool(self, x, p, stride=None):
        nonlocal part_index
        native = original_pool(x, p, stride)
        if x.shape[1] == module.width:
            if valid is None or part_index >= len(self.graph.part):
                raise ValueError('Unexpected anatomical pooling call')
            joints = self.graph.part[part_index]
            if x.shape[-1] != len(joints) or tuple(p) != (1, len(joints)) or stride is not None:
                raise ValueError('Native anatomical part layout changed')
            part_index += 1
            return native + module(x, valid[:, joints])
        return native

    encoder.forward = MethodType(forward, encoder)
    encoder.graph_max_pool = MethodType(pool, encoder)
    return dict(rank=16, width=256, centered=centered, parts=encoder.graph.part,
                parameters=sum(p.numel() for p in module.parameters()))


def attach_joint_bilinear(model, centered=False):
    return {s: attach_encoder_bilinear(getattr(model.signbert.embed, s), centered=centered)
            for s in ('st_gcn_hand', 'st_gcn_body')}
