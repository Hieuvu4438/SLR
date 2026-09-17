import torch
from .embedding_rollback_probe import KEYS, restore_tables


def test_restore_targets_and_reset_between_cells():
    core = torch.nn.Module()
    core.clip = torch.nn.Module()
    core.clip.positional_embedding = torch.nn.Parameter(torch.ones(2,2))
    core.clip.token_embedding = torch.nn.Embedding(2,2)
    core.other = torch.nn.Parameter(torch.tensor([7.]))
    trained = {k:torch.full((2,2),3.) for k in KEYS}
    initial = {k:torch.full((2,2),1.) for k in KEYS}
    restore_tables(core,trained,initial,(KEYS[0],))
    assert core.clip.positional_embedding.eq(1).all() and core.clip.token_embedding.weight.eq(3).all()
    restore_tables(core,trained,initial,(KEYS[1],))
    assert core.clip.positional_embedding.eq(3).all() and core.clip.token_embedding.weight.eq(1).all()
    assert core.other.item() == 7
