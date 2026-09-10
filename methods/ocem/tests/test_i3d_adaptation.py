from __future__ import annotations

import numpy as np
import pytest
import torch

from ocem.training.i3d_adaptation import (
    I3DAdaptationError,
    adaptation_update,
    apply_cico_train_augmentation,
    build_cico_sgd,
    sample_cico_augmentation_draws,
)


class _TinyI3D(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.logits = torch.nn.Linear(4, 2)

    def forward(self, values):
        return {"logits": self.logits(torch.tanh(self.encoder(values)))}


def test_augmentation_draws_are_deterministic_and_batch_shaped() -> None:
    first = sample_cico_augmentation_draws(2, seed=7)
    second = sample_cico_augmentation_draws(2, seed=7)
    assert first.horizontal_flip == second.horizontal_flip
    assert first.color_jitter == second.color_jitter
    assert np.array_equal(first.scale_uniform, second.scale_uniform)
    assert np.array_equal(first.offset_uniform, second.offset_uniform)
    assert first.scale_uniform.shape == (2, 1)
    assert first.offset_uniform.shape == (2, 2)


def test_augmentation_has_locked_shape_range_and_normalization() -> None:
    rgb = torch.linspace(0, 1, steps=2 * 3 * 16 * 32 * 32).reshape(2, 3, 16, 32, 32)
    draws = sample_cico_augmentation_draws(2, seed=0)
    output = apply_cico_train_augmentation(rgb, draws, input_resolution=16, resize_resolution=18)
    assert output.shape == (2, 3, 16, 16, 16)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= -0.5
    assert float(output.max()) <= 0.5


def test_sgd_groups_and_one_step_match_direct_cross_entropy() -> None:
    torch.manual_seed(0)
    port = _TinyI3D()
    reference = _TinyI3D()
    reference.load_state_dict(port.state_dict())
    port_optimizer = build_cico_sgd(port)
    reference_optimizer = build_cico_sgd(reference)
    assert [group["lr"] for group in port_optimizer.param_groups] == [0.01, 0.01]
    assert [group["momentum"] for group in port_optimizer.param_groups] == [0.9, 0.9]
    values = torch.tensor([[0.2, -0.1, 0.7], [0.4, 0.3, -0.2]])
    targets = torch.tensor([0, 1])
    loss, gradients = adaptation_update(port, port_optimizer, values, targets)
    reference_optimizer.zero_grad()
    reference_loss = torch.nn.functional.cross_entropy(
        reference(values)["logits"], targets, reduction="mean"
    )
    reference_loss.backward()
    reference_gradients = {
        name: parameter.grad.clone() for name, parameter in reference.named_parameters()
    }
    reference_optimizer.step()
    assert torch.equal(loss, reference_loss.detach())
    for name, parameter in port.named_parameters():
        assert torch.equal(gradients[name], reference_gradients[name])
        assert torch.equal(parameter, dict(reference.named_parameters())[name])


def test_invalid_augmentation_contract_fails_closed() -> None:
    draws = sample_cico_augmentation_draws(1, seed=0)
    with pytest.raises(I3DAdaptationError, match="expected"):
        apply_cico_train_augmentation(torch.zeros(1, 3, 8, 16, 16), draws)
