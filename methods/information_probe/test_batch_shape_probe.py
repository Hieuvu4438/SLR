import numpy as np
import pytest
import torch
from .batch_shape_probe import pad_rows, sensitivity


def test_padding_preserves_order_dtype_and_original():
    for dtype in (torch.bool, torch.int64, torch.float16, torch.float32):
        x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
        old = x.clone()
        y = pad_rows(x, 5)
        assert y.dtype == x.dtype and y.shape == (5, 2)
        assert torch.equal(y[:2], x) and torch.equal(y[2:], x[:1].expand(3, 2))
        y[2, 0] = 1
        assert torch.equal(x, old)
        assert pad_rows(x, 2) is x
    with pytest.raises(AssertionError):
        pad_rows(torch.empty(0, 2))


def test_rank_sensitivity_and_tie_entries():
    base = np.array([[2., 1.], [1., 2.]], dtype=np.float32)
    candidate = base.copy()
    candidate[1, 0] = 3.
    r = sensitivity(candidate, base)
    assert r['max_score_abs_delta'] == 2 and r['changed_score_n'] == 1
    assert r['directions']['T2V']['R1_lost_indices'] == [0]
    assert r['directions']['V2T']['R1_lost_indices'] == [1]
    assert sensitivity(np.ones((2, 2), dtype=np.float32), base)['T2V_expanded_entries'] == 4
