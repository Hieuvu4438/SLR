"""C22: length-normalized low-rank temporal modulation for pose/RGB tokens."""
from types import MethodType

import torch
from torch import nn


class SharedPhaseModulation(nn.Module):
    """Share a normalized temporal basis; keep stream channel factors separate."""

    def __init__(self, width=512, rank=3, anchors=32):
        super().__init__()
        if rank < 1 or anchors < 2 or width < 1:
            raise ValueError('Invalid phase-modulation dimensions')
        position = torch.linspace(0, 1, anchors).unsqueeze(1)
        frequency = torch.arange(1, rank + 1).unsqueeze(0)
        self.phase = nn.Parameter(torch.cos(torch.pi * position * frequency))
        self.scale = nn.ParameterDict({s:nn.Parameter(torch.zeros(rank,width)) for s in ('pose','rgb')})
        self.shift = nn.ParameterDict({s:nn.Parameter(torch.zeros(rank,width)) for s in ('pose','rgb')})
        self.rank, self.anchors = rank, anchors

    def _basis(self, padding_mask, dtype):
        if padding_mask.ndim != 2:
            raise ValueError('Expected B,T padding mask')
        valid = padding_mask == 0
        valid = valid.clone()
        valid[:,0] = False  # preserve native CLS exactly
        basis = self.phase.new_zeros((*valid.shape,self.rank))
        source = self.phase.T.unsqueeze(0)
        for index in range(valid.shape[0]):
            count = int(valid[index].sum())
            if count:
                interpolated = torch.nn.functional.interpolate(
                    source,size=count,mode='linear',align_corners=True).squeeze(0).T
                basis[index,valid[index]] = interpolated
        return basis.to(dtype=dtype),valid

    def apply_stream(self, x, padding_mask, stream):
        if stream not in self.scale or x.ndim != 3 or padding_mask.shape != x.shape[:2]:
            raise ValueError('Expected known stream B,T,D tokens and matching mask')
        basis,valid = self._basis(padding_mask,torch.float32)
        scale = torch.einsum('btr,rd->btd',basis,self.scale[stream])
        shift = torch.einsum('btr,rd->btd',basis,self.shift[stream])
        correction = x.float()*scale + shift
        correction = torch.where(valid[...,None],correction,torch.zeros_like(correction))
        return x + correction.to(x.dtype)

    def forward(self, pose, rgb, padding_mask):
        if pose.shape != rgb.shape:
            raise ValueError('Pose/RGB token shapes must match')
        return (self.apply_stream(pose,padding_mask,'pose'),
                self.apply_stream(rgb,padding_mask,'rgb'))


def attach_phase_modulation(model, rank=3, anchors=32):
    if hasattr(model,'temporal_modulation'):
        raise ValueError('Phase modulation already attached')
    width = model.clip.visual.proj.shape[-1]
    module = SharedPhaseModulation(width,rank,anchors).to(
        device=next(model.parameters()).device,dtype=torch.float32)
    model.add_module('temporal_modulation',module)
    original = model.get_visual_output

    def adapted(self,*args,**kwargs):
        mask,pose,rgb = original(*args,**kwargs)
        pose,rgb = self.temporal_modulation(pose,rgb,mask)
        return mask,pose,rgb

    model.get_visual_output = MethodType(adapted,model)
    return module


def configure_phase_modulation(model,active):
    for parameter in model.temporal_modulation.parameters():
        parameter.requires_grad_(True)
    model.temporal_modulation.train()
    active.append(model.temporal_modulation)
    return active
