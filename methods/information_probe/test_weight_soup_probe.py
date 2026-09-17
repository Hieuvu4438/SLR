import pytest
import torch

from .weight_soup_probe import average_states, state_digest


def test_weight_average_identity_dtype_and_buffers():
    a = {'x': torch.tensor([.1, .4], dtype=torch.float16), 'count': torch.tensor(3)}
    b = {'x': torch.tensor([.3, .8], dtype=torch.float16), 'count': torch.tensor(3)}
    repeat = average_states([a]*3)
    assert state_digest(repeat) == state_digest(a)
    value = average_states([a, b, a])
    assert torch.equal(value['x'], ((a['x'].double()*2+b['x'].double())/3).half())
    assert value['count'].item() == 3
    with pytest.raises(AssertionError):
        average_states([a, {**b, 'count': torch.tensor(4)}])
