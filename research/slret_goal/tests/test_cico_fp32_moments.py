import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cico_fp32_moments import FP32MomentAdamW
from adamw_arithmetic import isolated_step


SETTINGS=dict(lr=1e-5,betas=(.9,.98),eps=1e-6,weight_decay=.001)


def test_fp32_negative_control_multistep():
    a=torch.nn.Parameter(torch.tensor([.5,-.2]))
    b=torch.nn.Parameter(a.detach().clone())
    native=torch.optim.AdamW([a],foreach=False,**SETTINGS)
    fixed=FP32MomentAdamW([b],**SETTINGS)
    for step in range(4):
        a.grad=torch.tensor([.001*(step+1),-.002])
        b.grad=a.grad.clone()
        native.step(); fixed.step()
        assert torch.equal(a,b)
        for key in ['exp_avg','exp_avg_sq','step']:
            assert torch.equal(native.state[a][key],fixed.state[b][key])


def test_half_parameter_matches_registered_arithmetic():
    p=torch.nn.Parameter(torch.tensor([0.,.02],dtype=torch.float16))
    p.grad=torch.tensor([1e-4,-.003],dtype=torch.float16)
    expected=isolated_step(p,p.grad,SETTINGS,moment_fp32=True)
    opt=FP32MomentAdamW([p],**SETTINGS)
    opt.step()
    assert torch.equal(p,expected[0])
    assert torch.equal(opt.state[p]['exp_avg'],expected[1])
    assert torch.equal(opt.state[p]['exp_avg_sq'],expected[2])
    assert p.dtype==torch.float16


def test_resume_preserves_unrepresentable_fp32_moments():
    p=torch.nn.Parameter(torch.tensor([0.,.02],dtype=torch.float16))
    opt=FP32MomentAdamW([p],**SETTINGS)
    p.grad=torch.tensor([1e-4,-.003],dtype=torch.float16)
    opt.step()
    saved=copy.deepcopy(opt.state_dict())
    q=torch.nn.Parameter(p.detach().clone())
    resumed=FP32MomentAdamW([q],**SETTINGS)
    resumed.load_state_dict(saved)
    assert torch.equal(opt.state[p]['exp_avg_sq'],resumed.state[q]['exp_avg_sq'])
    assert resumed.state[q]['exp_avg_sq'].dtype==torch.float32
    q.grad=p.grad.clone()
    opt.step(); resumed.step()
    assert torch.equal(p,q)
    assert torch.equal(opt.state[p]['exp_avg_sq'],resumed.state[q]['exp_avg_sq'])
    bad=copy.deepcopy(saved)
    next(iter(bad['state'].values()))['exp_avg_sq']=torch.zeros(2,dtype=torch.float16)
    with pytest.raises(ValueError): resumed.load_state_dict(bad)
