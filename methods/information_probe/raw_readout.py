"""Crossed R2/R3 diagnostic modules; no new encoder pretraining."""
import torch
from torch import nn
from torch.nn import functional as F

from .scoring import Readout, channels, pooled


def raw_input(cache, raw_cache, arm, index, *, permutation=None):
    """Same [B,32,1024] interface and executed module for every information arm."""
    if arm == 'existing':
        # Retain token information: a pooled-only existing-feature control
        # would confound raw information with access to multiple tokens.
        v=cache['video_tokens'][index]
        vm=cache['video_mask'][index]==0
        selected=[]
        for row,valid in zip(v,vm):
            tokens=row[valid]
            if not len(tokens):
                raise ValueError('existing-feature control needs valid tokens')
            positions=torch.linspace(0,len(tokens)-1,32,device=v.device).long()
            selected.append(tokens[positions])
        tokens=torch.stack(selected)
        return torch.cat((tokens,tokens),-1)
    if arm == 'zero':
        return torch.zeros((len(index),32,1024),device=cache['video_tokens'].device)
    chosen=index if permutation is None else permutation[index]
    if arm in ('spatial','shuffled'):
        return raw_cache['spatial'][chosen].flatten(1,2)
    if arm == 'pooled':
        p=raw_cache['pooled'][chosen]
        return p[:,:,None,:].expand(-1,-1,4,-1).flatten(1,2)
    raise ValueError(arm)


class RawReadout(nn.Module):
    def __init__(self, regime, score_scale=1.):
        super().__init__()
        assert regime in ('R2','R3')
        self.regime=regime
        self.score_scale=float(score_scale)
        self.vproj=nn.Sequential(nn.Linear(1024,64),nn.GELU(),nn.Linear(64,64))
        self.tproj=nn.Sequential(nn.Linear(512,64),nn.GELU(),nn.Linear(64,64))
        if regime=='R2':
            self.scale=nn.Parameter(torch.zeros(2))
        else:
            self.existing=Readout('interaction')
            self.pair=nn.Sequential(nn.Linear(4*64+2,128),nn.GELU(),nn.Linear(128,2))
            nn.init.zeros_(self.pair[-1].weight)
            nn.init.zeros_(self.pair[-1].bias)

    def forward(self, raw, v, t, vm, tm):
        raw=self.vproj(F.normalize(raw,dim=-1))
        text=self.tproj(F.normalize(t,dim=-1))
        rawmask=torch.zeros(raw.shape[:2],device=raw.device,dtype=torch.long)
        a,b=channels(raw,text,rawmask,tm,1.,masked=True)
        if self.regime=='R2':
            return a*self.scale[0]*self.score_scale,b*self.scale[1]*self.score_scale
        left,right=torch.broadcast_tensors(raw.mean(1)[:,None],pooled(text,tm==1)[None])
        z=torch.cat((left,right,left*right,(left-right).abs(),torch.stack((a,b),-1)),-1)
        x,y=self.pair(z).unbind(-1)
        ex,ey=self.existing(v,t,vm,tm)
        return x+ex,y+ey
