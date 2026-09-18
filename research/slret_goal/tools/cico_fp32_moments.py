"""Conventional AdamW with FP32 moments and unchanged parameter storage dtype.

No master weights, new objective, or claimed optimization algorithm.
"""
import torch
from torch.optim.adamw import adamw


class FP32MomentAdamW(torch.optim.AdamW):
    def __init__(self,params,**kwargs):
        kwargs.update(foreach=False,fused=False)
        super().__init__(params,**kwargs)

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for group in self.param_groups:
            if any(group.get(k,False) for k in ['amsgrad','maximize','capturable','differentiable','foreach','fused']):
                raise ValueError('Unsupported optimizer mode for locked arithmetic control')
            parameters=[]
            grads=[]
            means=[]
            seconds=[]
            steps=[]
            for p in group['params']:
                if p.grad is None: continue
                if p.grad.is_sparse: raise ValueError('Sparse gradients unsupported')
                state=self.state[p]
                if not state:
                    state.update(step=torch.tensor(0.),exp_avg=torch.zeros_like(p,dtype=torch.float32),
                                 exp_avg_sq=torch.zeros_like(p,dtype=torch.float32))
                if state['exp_avg'].dtype!=torch.float32 or state['exp_avg_sq'].dtype!=torch.float32:
                    raise ValueError('Cannot repair already-rounded optimizer history')
                parameters.append(p)
                grads.append(p.grad.float())
                means.append(state['exp_avg'])
                seconds.append(state['exp_avg_sq'])
                steps.append(state['step'])
            adamw(parameters,grads,means,seconds,[],steps,foreach=False,capturable=False,
                  differentiable=False,fused=False,amsgrad=False,beta1=group['betas'][0],
                  beta2=group['betas'][1],lr=group['lr'],weight_decay=group['weight_decay'],
                  eps=group['eps'],maximize=False)
        return loss

    def load_state_dict(self,state_dict):
        # The standard loader casts moment state to parameter dtype. Preserve the
        # original FP32 tensors before that cast; never upcast rounded copies.
        for state in state_dict['state'].values():
            if any(state[k].dtype!=torch.float32 for k in ['exp_avg','exp_avg_sq']):
                raise ValueError('Saved moments are not FP32')
        super().load_state_dict(state_dict)
        for current,saved in zip(self.param_groups,state_dict['param_groups'],strict=True):
            for parameter,index in zip(current['params'],saved['params'],strict=True):
                if index not in state_dict['state']: continue
                for key in ['exp_avg','exp_avg_sq']:
                    self.state[parameter][key]=state_dict['state'][index][key].to(parameter.device,dtype=torch.float32).clone()
