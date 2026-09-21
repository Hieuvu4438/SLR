"""Pointwise hand/body interaction before SEDS temporal pooling (C03).

Native GCN_Embed concatenates [left, right, body], each of width512. This adds
low-rank multiplicative interactions at each frame; no query/rival conditioning,
temporal resampling, confidence weighting or cached contextual readout.
"""
from types import MethodType
import torch
from torch import nn


class ArticulatorInteraction(nn.Module):
    def __init__(self, width=512, rank=32, interaction='product'):
        super().__init__()
        if width < 1 or rank < 1 or interaction not in ('product','additive'):
            raise ValueError('Invalid articulator interaction configuration')
        self.width = width
        self.interaction = interaction
        self.norm = nn.LayerNorm(width,elementwise_affine=False)
        self.left = nn.Linear(width,rank,bias=False)
        self.right = nn.Linear(width,rank,bias=False)
        self.body = nn.Linear(width,rank,bias=False)
        self.output = nn.Linear(rank*3,width*3,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features):
        if features.ndim != 3 or features.shape[-1] != 3*self.width:
            raise ValueError('Expected [batch,time,3*articulator_width]')
        left,right,body = features.chunk(3,dim=-1)
        left = self.left(self.norm(left))
        right = self.right(self.norm(right))
        body = self.body(self.norm(body))
        if self.interaction == 'product':
            terms = [left*body,right*body,left*right]
        else:
            # Parameter-matched additive control, no multiplicative relations.
            terms = [(left+body)/2**.5,(right+body)/2**.5,(left+right)/2**.5]
        return features+self.output(torch.cat(terms,dim=-1))


def attach_articulator_interaction(signbert, width=512, rank=32, interaction='product'):
    """Preserve original parameter keys, attach before native temporal windows.

    Call after loading base checkpoint, before optimizer/DDP construction.
    Reconstruct this module before loading a candidate checkpoint with strict=True.
    """
    if hasattr(signbert,'articulator_interaction'):
        raise ValueError('Articulator interaction already attached')
    prototype = next(signbert.parameters())
    module = ArticulatorInteraction(width,rank,interaction).to(
        device=prototype.device,dtype=prototype.dtype)
    signbert.add_module('articulator_interaction',module)
    original = signbert.gcn_emb

    def adapted(self,pose):
        result = original(pose)
        result['feat'] = self.articulator_interaction(result['feat'])
        return result

    signbert.gcn_emb = MethodType(adapted,signbert)
    return module
