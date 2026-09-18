import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from adamw_arithmetic import isolated_step,normalized_update


GROUP=dict(lr=1e-5,betas=(.9,.98),weight_decay=.001,eps=1e-6)


def test_native_functional_matches_optimizer():
    for dtype in [torch.float16,torch.float32]:
        w=torch.tensor([.5,-.2,0.],dtype=dtype)
        g=torch.tensor([.01,-.003,.0001],dtype=dtype)
        expected=torch.nn.Parameter(w.clone())
        opt=torch.optim.AdamW([expected],foreach=False,**GROUP)
        expected.grad=g.clone()
        opt.step()
        p,m,v=isolated_step(w,g,GROUP)
        assert torch.equal(p,expected)
        assert torch.equal(m,opt.state[expected]['exp_avg'])
        assert torch.equal(v,opt.state[expected]['exp_avg_sq'])


def test_fixed_gradient_exposes_rounding_without_mutating_inputs():
    w=torch.tensor([0.,0.],dtype=torch.float16)
    g=torch.tensor([1e-4,2e-4],dtype=torch.float16)
    original=w.clone()
    low,lm,lv=isolated_step(w,g,GROUP)
    mixed,mm,mv=isolated_step(w,g,GROUP,moment_fp32=True)
    high,hm,hv=isolated_step(w,g,GROUP,master_fp32=True)
    assert torch.equal(w,original)
    assert torch.count_nonzero(lv)==0 and torch.all(hv>0)
    assert torch.equal(mm,hm) and torch.equal(mv,hv)
    assert not torch.equal(low,mixed)
    assert torch.equal(normalized_update(mm,mv,GROUP),normalized_update(hm,hv,GROUP))


def test_fp32_negative_control():
    w=torch.tensor([.5,-.2])
    g=torch.tensor([.001,-.005])
    arms=[isolated_step(w,g,GROUP,**kw) for kw in [{},{'moment_fp32':True},{'master_fp32':True}]]
    assert all(torch.equal(a,b) for arm in arms[1:] for a,b in zip(arms[0],arm))
