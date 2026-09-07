from __future__ import annotations

import torch
from torch import nn

from elsc.models.adapter import LocalResidualAdapter
from elsc.upstream.cico_bridge import CiCoBridge


class DummyCore(nn.Module):
    loose_type = True

    def get_visual_output(self, video, mask, shaped, video_frame, get_hidden):
        assert shaped and video_frame == 1 and get_hidden
        tokens = video.squeeze(-1).transpose(1, 2)
        cls = tokens.mean(1)
        return mask, tokens, cls

    def get_sequence_output(self, ids, segments, mask, shaped, get_hidden):
        tokens = torch.nn.functional.one_hot(ids, num_classes=8).float()
        return mask, tokens, tokens[:, -1]

    def get_similarity_logits(self, text, video, text_mask, video_mask, **kwargs):
        base = video.mean(1) @ text.mean(1).T
        return base, base + torch.arange(base.shape[0], device=base.device)[:, None] * 0.01, ()


def test_feature_bridge_shape_and_mask():
    bridge = CiCoBridge(DummyCore())
    h = torch.randn(2, 4, 8)
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    encoding = bridge.encode_video(h, valid)
    assert CiCoBridge.upstream_video(h).shape == (2, 8, 4, 1)
    assert encoding.tokens.shape == (2, 4, 8)
    assert encoding.mask.tolist() == [[1, 0, 0, 1, 1], [1, 0, 0, 0, 1]]


def test_zero_init_adapter_is_exact_parity_in_eval():
    adapter = LocalResidualAdapter(8, 4).eval()
    h = torch.randn(2, 4, 8)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    assert torch.equal(adapter(h, valid), h)


def test_mixed_score_does_not_transpose_t2i():
    i2t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    t2i = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    mixed = CiCoBridge.mixed_score(i2t, t2i, 0.25)
    assert torch.equal(mixed, 0.25 * i2t + 0.75 * t2i)


def test_paired_score_matches_aligned_all_pairs_entries():
    bridge = CiCoBridge(DummyCore())
    h = torch.randn(3, 4, 8)
    valid = torch.ones(3, 4, dtype=torch.bool)
    ids = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    mask = torch.ones_like(ids)
    video = bridge.encode_video(h, valid)
    text = bridge.encode_text(ids, torch.zeros_like(ids), mask)
    i2t, t2i = bridge.score(video, text, objective=True)
    expected = bridge.mixed_score(i2t, t2i, 0.3).diagonal()
    assert torch.equal(bridge.paired_score(video, text, dual_mix=0.3), expected)
