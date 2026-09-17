import torch
from .training_recipe_audit import apply_source_freeze, explicit_uncorrected


def test_actual_freeze_block_on_minimal_clip_names():
    model = torch.nn.Module()
    model.clip = torch.nn.Module()
    model.clip.token_embedding = torch.nn.Embedding(5, 2)
    model.clip.positional_embedding = torch.nn.Parameter(torch.ones(3, 2))
    model.clip.visual = torch.nn.Linear(2, 2)
    model.clip.ln_final = torch.nn.LayerNorm(2)
    apply_source_freeze(model)
    assert {n for n,p in model.named_parameters() if not p.requires_grad} == {'clip.token_embedding.weight','clip.positional_embedding'}


def test_uncorrected_first_step_not_bias_corrected():
    p,g = torch.tensor([1.],dtype=torch.float64),torch.tensor([.2],dtype=torch.float64)
    q,m,v = explicit_uncorrected(p,torch.zeros_like(p),torch.zeros_like(p),g,1e-4)
    expected = p-1e-4*(.1*g/((.02*g.square()).sqrt()+1e-6))
    assert torch.allclose(q,expected,atol=1e-14)
    assert float(p-q) < 1e-4
