import pytest
import torch
from .visual_input_inventory import input_digest, scoped_groups


def test_digest_includes_mask_dtype_and_numeric_values():
    h = torch.zeros(3, 2)
    valid = torch.tensor([True, True, False])
    original = input_digest(h, valid)
    assert original == input_digest(-h, valid)
    assert original != input_digest(h, ~valid)
    assert original != input_digest(h.half(), valid)
    h[0, 0] = 1
    assert original != input_digest(h, valid)
    h[0, 0] = torch.nan
    with pytest.raises(AssertionError):
        input_digest(h, valid)


def test_collision_scope_pair_counts():
    entries = [{'split': s, 'index': i, 'pair_id': str(i), 'digest': d}
               for i, (s, d) in enumerate([('train','x'),('train','x'),('dev','x'),('dev','y')])]
    summary, groups = scoped_groups(entries, 'digest')
    assert len(groups) == 1
    assert summary['train']['pair_n'] == 1
    assert summary['dev']['pair_n'] == 0
    assert summary['cross_split']['pair_n'] == 2
    assert summary['cross_split']['row_n'] == 3
