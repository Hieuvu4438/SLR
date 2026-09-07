from __future__ import annotations

import torch
from torch import nn

from elsc.data.tokenize import encode_cico_text
from elsc.losses.evidence import apply_input_intervention, evidence_losses
from elsc.models.retriever import ELSCRetriever
from elsc.train import _evidence_objective, _slice_text, _slice_video
from elsc.upstream.cico_bridge import TextEncoding


class TinyTokenizer:
    vocabulary = {
        "<|startoftext|>": 1,
        "pos": 2,
        "neg": 3,
        "<|endoftext|>": 7,
    }

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.split()

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocabulary[token] for token in tokens]


class TinyFilipCore(nn.Module):
    sim_header = "Filip"

    def __init__(self):
        super().__init__()
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(1.1))

    def get_visual_output(self, video, mask, shaped, video_frame, get_hidden):
        assert shaped and video_frame == 1 and get_hidden
        # CiCo applies projection, positional embeddings, and Transformer blocks
        # before scoring, so an input intervention does not emit a zero-norm token.
        tokens = video.squeeze(-1).transpose(1, 2) + 0.25
        return mask[:, 1:], tokens, tokens.mean(dim=1)

    def get_sequence_output(self, ids, segments, mask, shaped, get_hidden):
        del segments
        assert not shaped and get_hidden
        tokens = torch.nn.functional.one_hot(ids, num_classes=8).float()
        return mask, tokens, tokens.mean(dim=1)

    def get_similarity_logits(self, text, video, text_mask, video_mask, **kwargs):
        video_valid = video_mask == 0
        text_valid = text_mask == 1
        video = video / video.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        similarity = torch.einsum("afs,bts->abft", video, text)
        i2t_token = torch.nansum(
            similarity * torch.softmax(similarity / 0.07, dim=3), dim=3
        )
        i2t_token = i2t_token.masked_fill(~video_valid[:, None, :], 0.0)
        i2t = i2t_token.sum(dim=2) / video_valid.sum(dim=1)[:, None]
        t2i_token = torch.nansum(
            similarity * torch.softmax(similarity / 0.07, dim=2), dim=2
        )
        t2i_token = t2i_token.masked_fill(~text_valid[None, :, :], 0.0)
        t2i = t2i_token.sum(dim=2) / text_valid.sum(dim=1)[None, :]
        scale = self.clip.logit_scale.exp()
        return scale * i2t, scale * t2i, ()


def _legacy_margin(model, video, positive, negative, dual_mix):
    text = TextEncoding(
        torch.cat((positive.mask, negative.mask)),
        torch.cat((positive.tokens, negative.tokens)),
        torch.cat((positive.cls, negative.cls)),
    )
    i2t, t2i = model.bridge.score(video, text, objective=True)
    scores = model.bridge.mixed_score(i2t, t2i, dual_mix)[0]
    return (scores[0] - scores[1]) / model.core.clip.logit_scale.exp().detach()


def test_batched_checkpoint_ready_evidence_objective_matches_legacy_loop():
    torch.manual_seed(23)
    tokenizer = TinyTokenizer()
    model = ELSCRetriever(
        TinyFilipCore(),
        input_dim=8,
        hidden_dim=4,
        text_dim=8,
        core_frozen=True,
        adapter_enabled=True,
    )
    h = torch.randn(3, 4, 8)
    valid = torch.ones(3, 4, dtype=torch.bool)
    dense_index = torch.arange(4).expand(3, -1)
    positive_encoded = [encode_cico_text("pos", tokenizer, 4) for _ in range(3)]
    positive_inputs = tuple(
        torch.stack([item[index] for item in positive_encoded]) for index in range(3)
    )
    clean_video, _ = model.encode_video(h, valid)
    clean_text = model.encode_text(*positive_inputs)
    records = [
        [
            {
                "evidence_eligible": True,
                "pair_id": f"pair-{index}",
                "word_id": index,
                "evidence_remove_dense_indices": [0],
                "control_remove_dense_indices": [2],
                "negative_captions": ["neg"],
                "rho": 0.5 + 0.1 * index,
            }
        ]
        for index in range(3)
    ]
    config = {
        "seed": 42,
        "data": {"max_words": 4},
        "model": {"dual_mix": 0.3},
        "evidence": {
            "batch_fraction": 1.0,
            "dependence_margin": 10.0,
            "huber_delta": 0.05,
            "encoder_microbatch_size": 2,
            "score_microbatch_size": 2,
            "activation_checkpoint": False,
        },
    }

    dependence, invariance, count, diagnostics = _evidence_objective(
        model,
        tokenizer,
        h,
        valid,
        dense_index,
        clean_video,
        clean_text,
        records,
        config,
        torch.device("cpu"),
        epoch=0,
        step=0,
    )

    clean_margins = []
    evidence_margins = []
    control_margins = []
    for index in range(3):
        evidence_mask = torch.tensor([[True, False, False, False]])
        control_mask = torch.tensor([[False, False, True, False]])
        own_h = h[index : index + 1]
        own_valid = valid[index : index + 1]
        evidence_video, _ = model.encode_video(
            apply_input_intervention(own_h, evidence_mask, own_valid, 0.0), own_valid
        )
        control_video, _ = model.encode_video(
            apply_input_intervention(own_h, control_mask, own_valid, 0.0), own_valid
        )
        negative_input = encode_cico_text("neg", tokenizer, 4)
        negative = model.encode_text(*(value.unsqueeze(0) for value in negative_input))
        positive = _slice_text(clean_text, index)
        clean_margins.append(
            _legacy_margin(model, _slice_video(clean_video, index), positive, negative, 0.3)
        )
        evidence_margins.append(
            _legacy_margin(model, evidence_video, positive, negative, 0.3)
        )
        control_margins.append(
            _legacy_margin(model, control_video, positive, negative, 0.3)
        )
    expected_dependence, expected_invariance = evidence_losses(
        torch.stack(clean_margins),
        torch.stack(evidence_margins),
        torch.stack(control_margins),
        torch.tensor([0.5, 0.6, 0.7]),
        dependence_margin=10.0,
        huber_delta=0.05,
    )

    assert count == 3
    assert torch.allclose(dependence, expected_dependence, atol=2e-6, rtol=1e-6)
    assert torch.allclose(invariance, expected_invariance, atol=2e-6, rtol=1e-6)
    assert diagnostics["video_encoder_calls"] == 4
    assert diagnostics["paired_score_calls"] == 12
    assert diagnostics["negative_text_encoder_calls"] == 1
    (dependence + invariance).backward()
    assert model.adapter.up.weight.grad is not None
    assert torch.isfinite(model.adapter.up.weight.grad).all()
