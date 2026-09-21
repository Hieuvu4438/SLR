"""C13: layer-specific A+B joint graphs, borrowing the global 2s-AGCN term.

No sample-dependent attention, bone stream, or cross-articulator exchange.
Attach after loading native weights, before configuring trainable parameters.
The native parameter names and physical graph buffers remain unchanged.
"""
from types import MethodType

import torch
from torch import nn


def attach_graph_delta(convolution, adjacency):
    if hasattr(convolution, 'graph_delta'):
        raise ValueError('Adaptive graph already attached')
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError('Expected K x V x V adjacency')
    prototype = next(convolution.parameters())
    convolution.register_parameter('graph_delta', nn.Parameter(
        torch.zeros_like(adjacency, device=prototype.device, dtype=prototype.dtype)))
    original = convolution.forward

    def forward(self, x, graph):
        if graph.shape != self.graph_delta.shape:
            raise ValueError('Unexpected graph layout')
        output, _ = original(x, graph + self.graph_delta)
        # Return the physical graph, not a cumulative delta for later layers.
        return output, graph

    convolution.forward = MethodType(forward, convolution)


def attach_adaptive_graph(model):
    attached = []
    for stream in ('st_gcn_hand', 'st_gcn_body'):
        encoder = getattr(model.signbert.embed, stream)
        if len(encoder.st_gcn_networks) != 3:
            raise ValueError('Expected three native joint graph layers')
        for index, block in enumerate(encoder.st_gcn_networks):
            attach_graph_delta(block.gcn, encoder.A)
            attached.append(f'signbert.embed.{stream}.st_gcn_networks.{index}.gcn.graph_delta')
    return attached


def graph_statistics(model):
    return {n: dict(norm=float(p.detach().float().norm()),
                    max_abs=float(p.detach().abs().max()))
            for n, p in model.named_parameters() if n.endswith('.graph_delta')}
