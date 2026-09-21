"""C14 explicit bone geometry into first native joint GCN, retaining absolute XY.

Borrow bone-vector representation, not 2s-AGCN architecture. No adaptive graph,
new scorer, generic extra stream or learned missingness/confidence gate.
"""
from types import MethodType
import torch
from torch import nn


class BoneFeatures(nn.Module):
    def __init__(self, parents, width):
        super().__init__()
        self.register_buffer('parents', torch.tensor(parents, dtype=torch.long))
        self.output = nn.Conv2d(5, width, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def descriptors(self, xy):
        if xy.ndim != 4 or xy.shape[-2:] != (len(self.parents), 2):
            raise ValueError('Expected native B,T,V,2 pixel coordinates')
        if not torch.isfinite(xy).all():
            raise ValueError('Nonfinite native pose')
        parent = xy.index_select(-2, self.parents)
        valid = xy.ne(0).any(-1) & parent.ne(0).any(-1)
        # Root self-edge is zero; a missing endpoint must not create a fake bone.
        delta = (xy-parent).masked_fill(~valid[..., None], 0.)
        length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        # One-pixel denominator floor bounds direction for tiny/noisy bones.
        direction = delta / length.clamp_min(1.)
        return torch.cat((delta/256., length/256., direction), -1)

    def forward(self, xy):
        desc = self.descriptors(xy)
        b, t, v, _ = desc.shape
        return self.output(desc.reshape(b*t, v, 5).permute(0, 2, 1).unsqueeze(2))


def attach_encoder_bones(encoder):
    if hasattr(encoder, 'bone_features'):
        raise ValueError('Bone features already attached')
    graph = encoder.graph
    if graph.seqlen != 1:
        raise ValueError('Expected native framewise graph')
    parents = list(range(graph.num_node_each))
    for parent, child in graph.neighbour_link_all:
        if child == graph.center or parents[child] != child:
            raise ValueError('Expected directed anatomical tree')
        parents[child] = parent
    if sum(i==p for i,p in enumerate(parents)) != 1:
        raise ValueError('Incomplete bone tree')
    first = encoder.st_gcn_networks[0].gcn
    width = first.conv.out_channels // first.kernel_size
    prototype = next(encoder.parameters())
    module = BoneFeatures(parents, width).to(prototype.device, prototype.dtype)
    encoder.add_module('bone_features', module)
    original_encoder, original_graph = encoder.forward, first.forward
    geometry = None

    def encoder_forward(self, xy):
        nonlocal geometry
        if geometry is not None:
            raise RuntimeError('Reentrant bone encoder is unsupported')
        geometry = module(xy)
        try:
            return original_encoder(xy)
        finally:
            geometry = None

    def graph_forward(self, x, adjacency):
        output, graph_out = original_graph(x, adjacency)
        if geometry is None or geometry.shape != output.shape:
            raise ValueError('Bone/native first-GCN shape mismatch')
        return output + geometry, graph_out

    encoder.forward = MethodType(encoder_forward, encoder)
    first.forward = MethodType(graph_forward, first)
    return parents


def attach_bone_features(model):
    layouts = {}
    for stream in ('st_gcn_hand', 'st_gcn_body'):
        layouts[stream] = attach_encoder_bones(getattr(model.signbert.embed, stream))
    return layouts
