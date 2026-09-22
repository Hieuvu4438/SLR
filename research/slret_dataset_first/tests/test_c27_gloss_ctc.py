import pytest
import torch

from methods.gloss_bridge.ctc import GlossCTC


def test_gloss_ctc_finite_and_reaches_pose_tokens():
    head = GlossCTC(4, 3)
    pose = torch.randn(2, 7, 4, requires_grad=True)
    mask = torch.tensor([[0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 1, 1]])
    head.set_targets([[1, 1, 2], [2, 3]])
    loss = head(pose, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pose.grad).all()
    assert pose.grad[:, 1:].abs().sum() > 0
    assert pose.grad[:, 0].abs().sum() == 0  # CLS is not a CTC frame
    assert head.stats["eligible"] == 2


def test_gloss_ctc_rejects_ineligible_sequence():
    head = GlossCTC(4, 2)
    head.set_targets([[1, 1, 2]])
    with pytest.raises(ValueError, match="CTC-ineligible"):
        head(torch.randn(1, 4, 4), torch.tensor([[0, 0, 0, 0]]))


def test_gloss_ctc_rejects_missing_or_out_of_vocab_targets():
    head = GlossCTC(4, 2)
    with pytest.raises(ValueError, match="missing"):
        head(torch.randn(1, 5, 4), torch.zeros(1, 5, dtype=torch.long))
    head.set_targets([[3]])
    with pytest.raises(ValueError, match="outside TRAIN vocabulary"):
        head(torch.randn(1, 5, 4), torch.zeros(1, 5, dtype=torch.long))
