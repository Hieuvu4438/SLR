import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from methods.seds_adaptation.objectives import fused_priority_loss


def test_native_parity_and_changed_gradient():
    parts = [torch.tensor(float(i), requires_grad=True) for i in range(1, 7)]
    original = sum(parts)
    assert torch.equal(fused_priority_loss([original, *parts], 1), original)
    result = fused_priority_loss([original, *parts], .25)
    result.backward()
    assert [p.grad.item() for p in parts] == [1., .25, .25, 1., 1., 1.]


def test_disabled_losses_and_bad_weight():
    x = torch.tensor(2., requires_grad=True)
    assert fused_priority_loss([3*x, x, x, x, 0., 0., 0.], .25) == 1.5*x
    with pytest.raises(ValueError):
        fused_priority_loss([x]*7, float('nan'))
