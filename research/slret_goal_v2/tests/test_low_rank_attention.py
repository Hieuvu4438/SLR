import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.seds_adaptation.low_rank_attention import LowRankUpdate,attach_visual_lora,attach_text_lora


def test_half_weight_identity_fp32_parameters_and_updates():
    original=torch.randn(12,4,dtype=torch.float16)
    adapter=LowRankUpdate(original,2,2)
    assert torch.equal(adapter(original),original)
    assert adapter.A.dtype==adapter.B.dtype==torch.float32
    adapter(original).float().square().sum().backward()
    assert adapter.B.grad.abs().sum()>0
    assert torch.equal(adapter.A.grad,torch.zeros_like(adapter.A))


def fake_model(depth=1):
    model=nn.Module()
    for stream in ('clip','clip_rgb'):
        clip=nn.Module();clip.visual=nn.Module();clip.visual.transformer=nn.Module()
        blocks=[]
        for _ in range(depth):
            block=nn.Module();block.attn=nn.MultiheadAttention(8,2);blocks.append(block)
        clip.visual.transformer.resblocks=nn.ModuleList(blocks)
        setattr(model,stream,clip)
    return model


def test_three_upper_blocks_identity_gradient_and_frozen_base():
    model=fake_model(4)
    for p in model.parameters():
        p.requires_grad_(False)
    x=torch.randn(5,2,8)
    def forward():
        outputs=[]
        for stream in ('clip','clip_rgb'):
            y=x
            for block in getattr(model,stream).visual.transformer.resblocks:
                y=y+block.attn(y,y,y,need_weights=False)[0]
            outputs.append(y)
        return torch.stack(outputs)
    before=forward().detach()
    attached=attach_visual_lora(model,rank=2,alpha=2,upper_blocks=3)
    assert len(attached)==12 and torch.equal(forward(),before)
    forward().square().sum().backward()
    assert all(module.B.grad.abs().sum()>0 for _,module in attached)
    assert all(p.grad is None for n,p in model.named_parameters() if '.parametrizations.' not in n or n.endswith('original'))
    assert all('.0.' not in name for name,_ in attached)


def test_native_multihead_path_and_checkpoint_roundtrip():
    model=fake_model();x=torch.randn(5,2,8)
    attention=model.clip.visual.transformer.resblocks[0].attn
    before=attention(x,x,x,need_weights=False)[0].detach()
    attached=attach_visual_lora(model,rank=2,alpha=2)
    assert len(attached)==4
    assert torch.equal(attention(x,x,x,need_weights=False)[0],before)
    attention(x,x,x,need_weights=False)[0].square().sum().backward()
    assert attached[0][1].B.grad.abs().sum()>0
    assert attention.parametrizations.in_proj_weight.original.grad is None
    rebuilt=fake_model();attach_visual_lora(rebuilt,rank=2,alpha=2)
    rebuilt.load_state_dict(model.state_dict(),strict=True)
    other=rebuilt.clip.visual.transformer.resblocks[0].attn
    assert torch.equal(other(x,x,x,need_weights=False)[0],before)


def test_shared_text_lora_causal_mask_identity_and_roundtrip():
    import copy
    model=fake_model()
    model.clip.transformer=nn.Module()
    blocks=[]
    for _ in range(2):
        block=nn.Module();block.attn=nn.MultiheadAttention(8,2);blocks.append(block)
    model.clip.transformer.resblocks=nn.ModuleList(blocks)
    for p in model.parameters():
        p.requires_grad_(False)
    rebuilt=copy.deepcopy(model)
    x=torch.randn(5,2,8)
    mask=torch.triu(torch.full((5,5),float('-inf')),diagonal=1)
    attn=blocks[-1].attn
    before=attn(x,x,x,attn_mask=mask,need_weights=False)[0].detach()
    adapters=attach_text_lora(model,2,2)
    assert len(adapters)==2
    assert torch.equal(attn(x,x,x,attn_mask=mask,need_weights=False)[0],before)
    attn(x,x,x,attn_mask=mask,need_weights=False)[0].square().sum().backward()
    assert all(m.B.grad.abs().sum()>0 for _,m in adapters)
    with torch.no_grad():
        for _,m in adapters:
            m.B.add_(-.01*m.B.grad)
    assert not torch.equal(attn(x,x,x,attn_mask=mask,need_weights=False)[0],before)
    attach_text_lora(rebuilt,2,2)
    rebuilt.load_state_dict(model.state_dict(),strict=True)
    other=rebuilt.clip.transformer.resblocks[-1].attn
    assert torch.equal(other(x,x,x,attn_mask=mask,need_weights=False)[0],
                       attn(x,x,x,attn_mask=mask,need_weights=False)[0])
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
