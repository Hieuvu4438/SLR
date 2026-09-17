import numpy as np
import torch

from .scoring import channels
from .text_prefix_audit import bucket_credits, lcp, prefix_inventory, token_credits


def test_lcp_and_buckets():
    assert lcp((1, 2, 3), (1, 2, 4)) == 2
    assert lcp((1,), (1, 2)) == 1
    assert lcp((), (1,)) == 0
    value = np.arange(8, dtype=float)
    b = bucket_credits(value, 4, 2)
    assert b.tolist() == [0, 3, 7, 5, 13]
    assert b.sum() == value.sum()


def test_credit_conservation_including_inner_padding():
    torch.manual_seed(42)
    v, t = torch.randn(3, 6, 9), torch.randn(3, 5, 9)
    vm = torch.tensor([[1, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]])
    tm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    ab = channels(v, t, vm, tm, 100.)
    credit = token_credits(v, t, vm, tm, 100.)
    assert torch.allclose(credit.sum(-1), torch.stack([x.diagonal() for x in ab], 1), atol=2e-5, rtol=0)
    assert credit[0, 1, 3:].eq(0).all()
    assert credit[0, 0, 3:].ne(0).any()  # A includes inner padded text slots.


def test_prefix_inventory_excludes_whole_input_duplicates_only():
    sequences = [(1, 2), (1, 3), (4,), (4,)]
    tokens = torch.zeros(4, 4, 2)
    result = prefix_inventory(sequences, tokens)
    assert result['shared_prefix_rows'] == 2
    assert result['shared_prefix_token_slots'] == 2
    assert result['shared_prefix_groups'] == 1
    assert result['nonexact_shared_groups'] == 0
