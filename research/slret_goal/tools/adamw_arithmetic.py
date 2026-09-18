"""Isolated installed-PyTorch AdamW steps; not a training optimizer replacement."""
import torch
from torch.optim.adamw import adamw


@torch.no_grad()
def isolated_step(weight,gradient,group,*,moment_fp32=False,master_fp32=False,foreach=False):
    if weight.shape!=gradient.shape or not torch.isfinite(gradient).all():
        raise ValueError('Invalid fixed gradient')
    parameter=weight.detach().clone().float() if master_fp32 else weight.detach().clone()
    dtype=torch.float32 if moment_fp32 or master_fp32 else parameter.dtype
    grad=gradient.detach().to(dtype)
    m=torch.zeros_like(parameter,dtype=dtype)
    v=torch.zeros_like(parameter,dtype=dtype)
    step=torch.tensor(0.)
    adamw([parameter],[grad],[m],[v],[],[step],foreach=foreach,capturable=False,
          differentiable=False,fused=False,amsgrad=False,beta1=group['betas'][0],
          beta2=group['betas'][1],lr=group['lr'],weight_decay=group['weight_decay'],
          eps=group['eps'],maximize=False)
    return parameter,m,v


def normalized_update(m,v,group):
    return (m.float()/(1-group['betas'][0]))/(torch.sqrt(v.float()/(1-group['betas'][1]))+group['eps'])


def squares(tensor):
    return float(tensor.double().square().sum())
