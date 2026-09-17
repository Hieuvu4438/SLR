import torch

from .raw_readout import RawReadout,raw_input
from .test_scoring import inputs


def test_crossed_shape_and_train_only_permutation():
    v,t,vm,tm=inputs()
    cache={'video_tokens':v,'video_mask':vm}
    raw={'spatial':torch.randn(3,8,4,1024),'pooled':torch.randn(3,8,1024)}
    index=torch.arange(3)
    for arm in ('existing','zero','pooled','spatial','shuffled'):
        assert raw_input(cache,raw,arm,index).shape==(3,32,1024)
    perm=torch.tensor([1,2,0])
    torch.testing.assert_close(raw_input(cache,raw,'shuffled',index,permutation=perm),
                               raw['spatial'][perm].flatten(1,2))


def test_raw_zero_initialization():
    v,t,vm,tm=inputs()
    for regime in ('R2','R3'):
        model=RawReadout(regime)
        a,b=model(torch.randn(3,32,1024),v,t,vm,tm)
        assert torch.count_nonzero(a)==torch.count_nonzero(b)==0


def test_existing_control_keeps_tokens_not_just_mean():
    v,_,vm,_=inputs()
    cache={'video_tokens':v,'video_mask':vm}
    index=torch.arange(3)
    before=raw_input(cache,{},'existing',index)
    v[0,1]+=100
    v[0,2]-=100
    after=raw_input(cache,{},'existing',index)
    assert not torch.allclose(before,after)
    torch.testing.assert_close(after[...,:512],after[...,512:])
