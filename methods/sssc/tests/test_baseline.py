from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from method1.baseline import directional_scores_blocked, directional_scores_dense, encode_local


class ScalarHead(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[..., :1]


def _core():
    return SimpleNamespace(video_weight_fc=ScalarHead(), text_weight_fc=ScalarHead())


class LearnableCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_weight_fc = nn.Linear(3, 1)
        self.text_weight_fc = nn.Linear(3, 1)


def test_corrected_score_is_padding_invariant() -> None:
    core = _core()
    video = torch.tensor(
        [[[9.0, 9.0], [1.0, 0.0], [0.0, 1.0], [50.0, -50.0]]], requires_grad=True
    )
    video_ignore = torch.tensor([[True, False, False, True]])
    text = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [99.0, -10.0]]], requires_grad=True)
    text_valid = torch.tensor([[True, True, False]])
    first = directional_scores_dense(core, video, video_ignore, text, text_valid)
    changed_video = video.detach().clone()
    changed_video[:, 0] = torch.tensor([-1000.0, 2000.0])
    changed_video[:, 3] = torch.tensor([5000.0, 9000.0])
    changed_text = text.detach().clone()
    changed_text[:, 2] = torch.tensor([-9000.0, 3000.0])
    second = directional_scores_dense(
        core, changed_video, video_ignore, changed_text, text_valid
    )
    torch.testing.assert_close(first[0], second[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first[1], second[1], atol=1e-6, rtol=1e-6)


def test_dense_and_blocked_scores_and_gradients_match() -> None:
    generator = torch.Generator().manual_seed(7)
    video = torch.randn(3, 4, 5, generator=generator, requires_grad=True)
    text = torch.randn(3, 3, 5, generator=generator, requires_grad=True)
    video_ignore = torch.tensor(
        [[True, False, False, True], [True, False, False, False], [True, False, True, True]]
    )
    text_valid = torch.tensor([[True, True, False], [True, True, True], [True, False, False]])
    dense = directional_scores_dense(_core(), video, video_ignore, text, text_valid)
    dense_loss = dense[0].sum() + dense[1].sum()
    dense_gradients = torch.autograd.grad(dense_loss, (video, text), retain_graph=True)
    blocked = directional_scores_blocked(
        _core(), video, video_ignore, text, text_valid, video_block=2, text_block=2
    )
    blocked_loss = blocked[0].sum() + blocked[1].sum()
    blocked_gradients = torch.autograd.grad(blocked_loss, (video, text))
    checkpointed = directional_scores_blocked(
        _core(),
        video,
        video_ignore,
        text,
        text_valid,
        video_block=2,
        text_block=2,
        checkpoint_blocks=True,
    )
    checkpointed_gradients = torch.autograd.grad(
        checkpointed[0].sum() + checkpointed[1].sum(), (video, text)
    )
    torch.testing.assert_close(dense[0], blocked[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense[1], blocked[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense_gradients[0], blocked_gradients[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dense_gradients[1], blocked_gradients[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dense[0], checkpointed[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense[1], checkpointed[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        dense_gradients[0], checkpointed_gradients[0], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        dense_gradients[1], checkpointed_gradients[1], atol=1e-5, rtol=1e-5
    )


def test_encode_local_unpacks_nine_value_contract() -> None:
    batch = 2
    dim = 4
    text_positions = 3
    video_positions = 65
    text_valid = torch.tensor([[True, True, False], [True, True, True]])
    video_ignore = torch.ones(batch, video_positions, dtype=torch.bool)
    video_ignore[:, 1:3] = False
    values = (
        torch.randn(batch, text_positions, dim),
        text_valid,
        torch.randn(batch, dim),
        torch.randn(batch, video_positions, dim),
        video_ignore,
        torch.randn(batch, dim),
        torch.randn(batch, text_positions, dim),
        text_valid,
        torch.randn(batch, dim),
    )

    class Student:
        def get_text_video_feat(self, *args, **kwargs):
            assert kwargs["shaped"] is True
            assert kwargs["video_frame"] == 1
            return values

    batch_value = {
        "input_ids": torch.zeros(batch, text_positions, dtype=torch.long),
        "token_type_ids": torch.zeros(batch, text_positions, dtype=torch.long),
        "text_valid": text_valid,
        "input_ids_aug": torch.zeros(batch, text_positions, dtype=torch.long),
        "text_aug_valid": text_valid,
        "video_features": torch.zeros(batch, 1024, 64, 1),
        "video_ignore_raw": video_ignore,
    }
    encoding = encode_local(Student(), batch_value)
    assert encoding.video_raw is values[3]
    assert encoding.text_aug_raw is values[6]


def test_checkpointed_blocks_match_weighting_head_parameter_gradients() -> None:
    torch.manual_seed(19)
    dense_core = LearnableCore()
    blocked_core = LearnableCore()
    blocked_core.load_state_dict(dense_core.state_dict())
    video_dense = torch.randn(3, 4, 3, requires_grad=True)
    text_dense = torch.randn(3, 3, 3, requires_grad=True)
    video_blocked = video_dense.detach().clone().requires_grad_(True)
    text_blocked = text_dense.detach().clone().requires_grad_(True)
    video_ignore = torch.tensor(
        [[True, False, False, True], [True, False, False, False], [True, False, True, True]]
    )
    text_valid = torch.tensor(
        [[True, True, False], [True, True, True], [True, False, False]]
    )
    dense = directional_scores_dense(
        dense_core, video_dense, video_ignore, text_dense, text_valid
    )
    (dense[0].sum() + dense[1].sum()).backward()
    blocked = directional_scores_blocked(
        blocked_core,
        video_blocked,
        video_ignore,
        text_blocked,
        text_valid,
        video_block=2,
        text_block=2,
        checkpoint_blocks=True,
    )
    (blocked[0].sum() + blocked[1].sum()).backward()
    torch.testing.assert_close(dense[0], blocked[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dense[1], blocked[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(video_dense.grad, video_blocked.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(text_dense.grad, text_blocked.grad, atol=1e-5, rtol=1e-5)
    for (_, dense_parameter), (_, blocked_parameter) in zip(
        dense_core.named_parameters(), blocked_core.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            dense_parameter.grad, blocked_parameter.grad, atol=1e-5, rtol=1e-5
        )
