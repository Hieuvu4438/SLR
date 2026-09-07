from __future__ import annotations

import torch

from elsc.data.cache_dataset import lexical_tensors_from_batch
from elsc.losses.lexical import lexical_loss
from elsc.models.adapter import LocalHead, LocalResidualAdapter
from elsc.models.retriever import ELSCRetriever
from elsc.train import _auxiliary_gradient_diagnostic, _checkpointed_video_batches


class IdentityVideoCore(torch.nn.Module):
    def get_visual_output(self, video, mask, shaped, video_frame, get_hidden):
        assert shaped and video_frame == 1 and get_hidden
        tokens = video.squeeze(-1).transpose(1, 2)
        return mask[:, 1:], tokens, tokens.mean(dim=1)


def test_zero_init_gradient_reaches_up_then_down_after_update():
    adapter = LocalResidualAdapter(8, 4)
    head = LocalHead(8, 4)
    optimizer = torch.optim.SGD([*adapter.parameters(), *head.parameters()], lr=0.1)
    h = torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    first = head(adapter(h, valid))[..., 0].sum()
    first.backward()
    assert adapter.up.weight.grad is not None and adapter.up.weight.grad.norm() > 0
    assert adapter.down.weight.grad is not None and adapter.down.weight.grad.norm() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = head(adapter(h, valid))[..., 1].sum()
    second.backward()
    assert adapter.down.weight.grad is not None and adapter.down.weight.grad.norm() > 0


def test_raw_locality_and_padding_residual():
    adapter = LocalResidualAdapter(8, 4)
    with torch.no_grad():
        adapter.up.weight.normal_()
    h = torch.randn(1, 4, 8)
    valid = torch.tensor([[True, True, True, False]])
    before = adapter(h, valid)
    changed = h.clone()
    changed[:, 1] += 100
    after = adapter(changed, valid)
    assert torch.equal(before[:, 0], after[:, 0])
    assert torch.equal(before[:, 2], after[:, 2])
    assert torch.equal(before[:, 3], h[:, 3])


def test_empty_cache_batch_keeps_zero_gradient_graph_for_head_only_control():
    head = LocalHead(8, 4)
    h = torch.randn(2, 3, 8)
    z = head(h)
    tensors = lexical_tensors_from_batch(
        z,
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        [[], []],
        torch.randn(5, 4),
    )
    value, count = lexical_loss(*tensors[:6])
    value.backward()
    assert count.item() == 0
    assert head.proj.weight.grad is not None
    assert head.proj.weight.grad.norm() == 0


def test_auxiliary_gradient_diagnostic_accepts_frozen_adapter_control():
    adapter = LocalResidualAdapter(8, 4).requires_grad_(False)
    head = LocalHead(8, 4)
    h = torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    loss = head(adapter(h, valid)).sum()
    diagnostics = _auxiliary_gradient_diagnostic(
        loss,
        [
            ("adapter_up", adapter.up.weight),
            ("adapter_down", adapter.down.weight),
            ("local_head", head.proj.weight),
        ],
    )
    assert diagnostics["adapter_up"] == 0.0
    assert diagnostics["adapter_down"] == 0.0
    assert diagnostics["local_head"] > 0.0


def test_checkpointed_video_microbatches_preserve_values_and_adapter_gradient():
    torch.manual_seed(19)
    model = ELSCRetriever(
        IdentityVideoCore(),
        input_dim=8,
        hidden_dim=4,
        text_dim=8,
        core_frozen=True,
        adapter_enabled=True,
    )
    h = torch.randn(5, 3, 8)
    valid = torch.ones(5, 3, dtype=torch.bool)
    direct, _ = model.encode_video(h, valid)
    checkpointed, calls = _checkpointed_video_batches(
        model,
        h,
        valid,
        microbatch_size=2,
        activation_checkpoint=True,
    )
    assert calls == 3
    assert torch.equal(checkpointed.mask, direct.mask)
    assert torch.equal(checkpointed.tokens, direct.tokens)
    assert torch.equal(checkpointed.cls, direct.cls)
    checkpointed.tokens.sum().backward()
    assert model.adapter.up.weight.grad is not None
    assert model.adapter.up.weight.grad.norm() > 0
