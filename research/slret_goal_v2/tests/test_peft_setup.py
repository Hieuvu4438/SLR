import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.seds_adaptation.peft_setup import configure_lora_fusion,configure_peft_learning_rates,enable_gcn_adaptation
from methods.seds_adaptation.train_policies import training_modes


def test_peft_trainable_groups_and_lr_partition():
    model=nn.Module();model.fusion=nn.Linear(8,8);model.signbert=nn.BatchNorm1d(8)
    for stream in ('clip','clip_rgb'):
        clip=nn.Module();clip.visual=nn.Module();clip.visual.transformer=nn.Module()
        block=nn.Module();block.attn=nn.MultiheadAttention(8,2)
        clip.visual.transformer.resblocks=nn.ModuleList([block]);setattr(model,stream,clip)
    active=configure_lora_fusion(model,rank=2,alpha=2)
    assert len(active)==5 and not model.signbert.training
    trainable={name for name,p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith('fusion.') or '.parametrizations.' in name for name in trainable)
    assert not any(name.endswith('original') for name in trainable)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.001)
    configure_peft_learning_rates(optimizer,model,1e-4,1e-5)
    assert {g['lr'] for g in optimizer.param_groups}=={1e-4,1e-5}
    assert all(g['weight_decay']==.001 for g in optimizer.param_groups)
    assert sum(len(g['params']) for g in optimizer.param_groups)==len(trainable)


def test_gcn_lora_composition_updates_and_frozen_statistics():
    model=nn.Module(); model.fusion=nn.Linear(8,8); model.signbert=nn.Module()
    model.signbert.embed=nn.Sequential(nn.Linear(8,8),nn.BatchNorm1d(8),nn.Dropout(.5))
    for stream in ('clip','clip_rgb'):
        clip=nn.Module(); clip.visual=nn.Module(); clip.visual.transformer=nn.Module()
        block=nn.Module(); block.attn=nn.MultiheadAttention(8,2,batch_first=True)
        clip.visual.transformer.resblocks=nn.ModuleList([block]); setattr(model,stream,clip)
    active=configure_lora_fusion(model,rank=2,alpha=2)
    enable_gcn_adaptation(model)
    optimizer=torch.optim.AdamW(model.parameters(),weight_decay=.001)
    groups=configure_peft_learning_rates(optimizer,model,1e-4,1e-5,encoder_lr=1e-6)
    assert {g['adaptation_group']:g['lr'] for g in groups}=={'encoder':1e-6,'fusion':1e-5,'lora':1e-4}
    buffers={n:b.clone() for n,b in model.named_buffers()}
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    for _ in range(2):
        training_modes(model,active); optimizer.zero_grad()
        x=model.signbert.embed(torch.randn(4,8)).unsqueeze(1)
        outputs=[]
        for stream in ('clip','clip_rgb'):
            attn=getattr(model,stream).visual.transformer.resblocks[0].attn
            outputs.append(attn(x,x,x)[0])
        model.fusion(sum(outputs)).square().mean().backward(); optimizer.step()
    assert not model.signbert.embed.training
    assert all(torch.equal(buffers[n],b) for n,b in model.named_buffers())
    changed=[n for n,p in model.named_parameters() if not torch.equal(p,before[n])]
    assert any(n.startswith('signbert.embed.') for n in changed)
    assert any(n.startswith('fusion.') for n in changed)
    assert sum('.parametrizations.' in n and n.endswith('.B') for n in changed)==4
    assert not any(n.endswith('original') for n in changed)
