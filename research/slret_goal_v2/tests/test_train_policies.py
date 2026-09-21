import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.seds_adaptation.train_policies import configure_trainable_stage, training_modes


def test_stages_freeze_lower_weights_and_batchnorm():
    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer=nn.Module();self.transformer.resblocks=nn.Sequential(nn.Linear(4,4),nn.Linear(4,4))
            self.ln_post=nn.LayerNorm(4);self.proj=nn.Parameter(torch.randn(4,4))
    model=nn.Module();model.fusion=nn.Linear(4,4);model.signbert=nn.BatchNorm1d(4)
    model.clip=nn.Module();model.clip.visual=Visual();model.clip.text=nn.Linear(4,4)
    model.clip_rgb=nn.Module();model.clip_rgb.visual=Visual()
    configure_trainable_stage(model,'fusion')
    assert all(p.requires_grad == name.startswith('fusion.') for name,p in model.named_parameters())
    assert model.training and model.fusion.training and not model.signbert.training
    active=configure_trainable_stage(model,'upper')
    assert model.clip.visual.transformer.resblocks[1].weight.requires_grad
    assert not model.clip.visual.transformer.resblocks[0].weight.requires_grad
    assert not model.clip.text.weight.requires_grad
    assert not model.signbert.weight.requires_grad
    assert not model.clip.visual.proj.requires_grad
    model.eval();training_modes(model,active)
    assert model.training and not model.signbert.training
