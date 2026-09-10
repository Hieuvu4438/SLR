from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ocem.baselines.cico_adapter import CiCoAdapter, CiCoAdapterError, CiCoEncoded


class _TinyCiCo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip = SimpleNamespace(
            logit_scale=torch.nn.Parameter(torch.tensor(1.25)),
            visual=SimpleNamespace(feature_len=3),
        )
        self.dual_mix = 0.3
        self.mix_design = "balance"
        self.text_projection = torch.nn.Linear(4, 4, bias=False)
        self.video_projection = torch.nn.Linear(4, 4, bias=False)

    def get_sequence_output(self, ids, token_types, attention_mask, shaped=True):
        del token_types, attention_mask, shaped
        tokens = torch.nn.functional.one_hot(ids, num_classes=4).float()
        tokens = self.text_projection(tokens)
        lengths = ids.argmax(dim=-1)
        mask = torch.ones(ids.shape, dtype=torch.long, device=ids.device)
        for row, length in enumerate(lengths):
            mask[row, length + 1 :] = 0
        return mask, tokens, tokens[:, -1]

    def get_visual_output(self, video, video_mask, shaped=True, video_frame=1):
        del shaped, video_frame
        local = video.squeeze(-1).transpose(1, 2)
        cls = torch.zeros(local.shape[0], 1, local.shape[-1], device=local.device)
        tokens = self.video_projection(torch.cat((cls, local), dim=1))
        return video_mask, tokens, tokens[:, 0]


def _encoded_fixture() -> CiCoEncoded:
    visual = torch.tensor(
        [
            [[0.1, 0.2], [0.7, 0.3], [0.2, 0.8]],
            [[0.3, 0.1], [0.4, 0.6], [0.9, 0.1]],
        ],
        requires_grad=True,
    )
    text = torch.tensor(
        [
            [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]],
            [[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]],
        ],
        requires_grad=True,
    )
    augmented = (text.detach() + torch.tensor([0.03, -0.02])).requires_grad_()
    return CiCoEncoded(
        visual_tokens=visual,
        canonical_text_tokens=text,
        augmented_text_tokens=augmented,
        upstream_video_mask=torch.tensor([[1, 0, 0], [1, 0, 1]]),
        canonical_text_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        augmented_text_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )


def _direct_upstream_scores(encoded: CiCoEncoded, scale: torch.Tensor):
    visual = encoded.visual_tokens / encoded.visual_tokens.norm(dim=-1, keepdim=True)
    text = encoded.canonical_text_tokens / encoded.canonical_text_tokens.norm(
        dim=-1, keepdim=True
    )
    augmented = encoded.augmented_text_tokens / encoded.augmented_text_tokens.norm(
        dim=-1, keepdim=True
    )
    affinity = torch.einsum("ais,bjs->abij", visual, text)
    affinity_aug = torch.einsum("ais,bjs->abij", visual, augmented)
    after_i2t = torch.nansum(affinity * torch.softmax(affinity / 0.07, dim=3), dim=3)
    video_valid = (encoded.upstream_video_mask == 0).unsqueeze(1).repeat(1, 2, 1)
    after_i2t[~video_valid] = 0
    i2t = scale * torch.nansum(after_i2t, dim=-1) / video_valid.sum(dim=-1)
    after_t2i = torch.nansum(
        affinity_aug * torch.softmax(affinity_aug / 0.07, dim=2), dim=2
    )
    text_valid = (encoded.augmented_text_mask == 1).unsqueeze(0).repeat(2, 1, 1)
    after_t2i[~text_valid] = 0
    t2i = scale * torch.nansum(after_t2i * text_valid, dim=-1) / text_valid.sum(dim=-1)
    return i2t, t2i


def _cross_en(scores: torch.Tensor) -> torch.Tensor:
    return -torch.diagonal(torch.log_softmax(scores, dim=-1)).mean()


def test_mask_and_tensor_mapping_match_upstream_contract() -> None:
    valid = torch.tensor([[True, False, True], [False, True, True]])
    mask = CiCoAdapter.upstream_video_mask(valid)
    assert torch.equal(mask, torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]]))
    local = torch.arange(2 * 3 * 1024).reshape(2, 3, 1024)
    video = CiCoAdapter.upstream_video_tensor(local)
    assert video.shape == (2, 1024, 3, 1)
    assert torch.equal(video[:, :, :, 0].transpose(1, 2), local)


def test_encode_exposes_internal_true_valid_masks() -> None:
    torch.manual_seed(0)
    model = _TinyCiCo()
    adapter = CiCoAdapter(model)
    ids = torch.tensor([[0, 1, 3], [2, 3, 0]])
    batch = {
        "local_h": torch.randn(2, 3, 1024)[:, :, :4],
        "video_valid": torch.tensor([[True, True, False], [True, True, True]]),
        "text_ids": ids,
        "text_input_valid": torch.ones_like(ids, dtype=torch.bool),
    }
    # Tiny fixture has four feature dimensions; exercise encode after padding to contract width.
    batch["local_h"] = torch.nn.functional.pad(batch["local_h"], (0, 1020))
    model.video_projection = torch.nn.Linear(1024, 4, bias=False)
    encoded = adapter.encode(batch, training=False)
    assert encoded.Z.shape == (2, 3, 4)
    assert torch.equal(encoded.video_valid, batch["video_valid"])
    assert encoded.Y.shape == (2, 3, 4)
    assert model.training is False


def test_raw_scores_scaled_scores_loss_gradients_and_legacy_mix_match_source() -> None:
    model = _TinyCiCo()
    adapter = CiCoAdapter(model)
    encoded = _encoded_fixture()
    direct_i2t, direct_t2i = _direct_upstream_scores(encoded, adapter.logit_scale())
    raw = adapter.directional_matrices(encoded)
    assert torch.allclose(adapter.logit_scale() * raw.raw_t2v, direct_i2t, atol=1e-6)
    assert torch.allclose(adapter.logit_scale() * raw.raw_v2t, direct_t2i, atol=1e-6)

    direct_loss = (
        0.3 * _cross_en(direct_i2t)
        + 0.7 * _cross_en(direct_i2t.T)
        + 0.3 * _cross_en(direct_t2i.T)
        + 0.7 * _cross_en(direct_t2i)
    ) / 2
    adapter_loss = adapter.original_training_loss(encoded)
    direct_grad = torch.autograd.grad(
        direct_loss,
        (encoded.visual_tokens, encoded.canonical_text_tokens, encoded.augmented_text_tokens),
        retain_graph=True,
    )
    adapter_grad = torch.autograd.grad(
        adapter_loss,
        (encoded.visual_tokens, encoded.canonical_text_tokens, encoded.augmented_text_tokens),
    )
    assert torch.allclose(adapter_loss, direct_loss, atol=1e-6)
    for actual, expected in zip(adapter_grad, direct_grad, strict=True):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    legacy = adapter.legacy_eval_matrices(encoded)
    expected_mix = 0.3 * direct_i2t + 0.7 * direct_t2i
    assert torch.allclose(legacy.logits_t2v, expected_mix, atol=1e-6)
    assert torch.equal(legacy.logits_t2v, legacy.logits_v2t)


def test_one_optimizer_update_matches_independent_upstream_formula() -> None:
    model_adapter = _TinyCiCo()
    model_direct = _TinyCiCo()
    model_direct.load_state_dict(model_adapter.state_dict())
    adapter = CiCoAdapter(model_adapter)
    encoded_adapter = _encoded_fixture()
    encoded_direct = CiCoEncoded(
        visual_tokens=encoded_adapter.visual_tokens.detach().clone(),
        canonical_text_tokens=encoded_adapter.canonical_text_tokens.detach().clone(),
        augmented_text_tokens=encoded_adapter.augmented_text_tokens.detach().clone(),
        upstream_video_mask=encoded_adapter.upstream_video_mask,
        canonical_text_mask=encoded_adapter.canonical_text_mask,
        augmented_text_mask=encoded_adapter.augmented_text_mask,
    )
    adapter_parameter = torch.nn.Parameter(encoded_adapter.visual_tokens.detach().clone())
    direct_parameter = torch.nn.Parameter(encoded_direct.visual_tokens.detach().clone())
    encoded_adapter = CiCoEncoded(
        visual_tokens=adapter_parameter,
        canonical_text_tokens=encoded_adapter.canonical_text_tokens.detach(),
        augmented_text_tokens=encoded_adapter.augmented_text_tokens.detach(),
        upstream_video_mask=encoded_adapter.upstream_video_mask,
        canonical_text_mask=encoded_adapter.canonical_text_mask,
        augmented_text_mask=encoded_adapter.augmented_text_mask,
    )
    encoded_direct = CiCoEncoded(
        visual_tokens=direct_parameter,
        canonical_text_tokens=encoded_direct.canonical_text_tokens,
        augmented_text_tokens=encoded_direct.augmented_text_tokens,
        upstream_video_mask=encoded_direct.upstream_video_mask,
        canonical_text_mask=encoded_direct.canonical_text_mask,
        augmented_text_mask=encoded_direct.augmented_text_mask,
    )
    optimizer_adapter = torch.optim.SGD([adapter_parameter], lr=0.02, momentum=0.9)
    optimizer_direct = torch.optim.SGD([direct_parameter], lr=0.02, momentum=0.9)

    optimizer_adapter.zero_grad()
    adapter.original_training_loss(encoded_adapter).backward()
    optimizer_adapter.step()
    optimizer_direct.zero_grad()
    direct_i2t, direct_t2i = _direct_upstream_scores(
        encoded_direct, model_direct.clip.logit_scale.exp()
    )
    direct_loss = (
        0.3 * _cross_en(direct_i2t)
        + 0.7 * _cross_en(direct_i2t.T)
        + 0.3 * _cross_en(direct_t2i.T)
        + 0.7 * _cross_en(direct_t2i)
    ) / 2
    direct_loss.backward()
    optimizer_direct.step()
    assert torch.allclose(adapter_parameter, direct_parameter, atol=1e-7, rtol=1e-6)


def test_pair_orientation_is_explicit_and_asymmetric() -> None:
    adapter = CiCoAdapter(_TinyCiCo())
    encoded = _encoded_fixture()
    matrices = adapter.directional_matrices(encoded)
    t2v, v2t = adapter.directional_scores(encoded, [(1, 0), (0, 1)])
    assert torch.equal(t2v, torch.stack((matrices.raw_t2v[1, 0], matrices.raw_t2v[0, 1])))
    assert torch.equal(v2t, torch.stack((matrices.raw_v2t[1, 0], matrices.raw_v2t[0, 1])))
    assert not torch.equal(t2v, v2t)


def test_invalid_contract_fails_closed() -> None:
    adapter = CiCoAdapter(_TinyCiCo())
    with pytest.raises(CiCoAdapterError, match="bool"):
        adapter.upstream_video_mask(torch.ones(1, 2))
    with pytest.raises(CiCoAdapterError, match="square"):
        adapter.original_training_loss(
            CiCoEncoded(
                visual_tokens=torch.randn(2, 2, 3),
                canonical_text_tokens=torch.randn(3, 2, 3),
                augmented_text_tokens=torch.randn(3, 2, 3),
                upstream_video_mask=torch.tensor([[1, 0], [1, 0]]),
                canonical_text_mask=torch.ones(3, 2, dtype=torch.long),
                augmented_text_mask=torch.ones(3, 2, dtype=torch.long),
            )
        )
