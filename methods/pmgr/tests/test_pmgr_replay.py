from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from pmgr.losses import objective_loss
from pmgr.replay import two_level_backward
from pmgr.scoring import EncodedPMGRBatch, mixed_pair_scores


class TinyRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 3, bias=False, dtype=torch.float64)
        self.dropout = nn.Dropout(0.2)
        self.logit_scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))

    def encode_video(self, features, padding_mask):
        hidden = self.dropout(self.shared(features))
        return hidden, padding_mask.eq(0)

    def encode_text(self, features, segments, valid):
        del segments
        hidden = self.dropout(self.shared(features))
        return hidden, valid.bool()


def _batch():
    generator = torch.Generator().manual_seed(11)
    return {
        "video_features": torch.randn(5, 4, 4, generator=generator, dtype=torch.float64),
        "video_padding_mask": torch.tensor(
            [[0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]]
        ),
        "input_ids": torch.randn(3, 5, 4, generator=generator, dtype=torch.float64),
        "segment_ids": torch.zeros(3, 5),
        "input_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 1, 0]]),
        "aug_input_ids": torch.randn(3, 6, 4, generator=generator, dtype=torch.float64),
        "aug_segment_ids": torch.zeros(3, 6),
        "aug_input_mask": torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 0]]),
        "video_to_group": torch.tensor([0, 0, 1, 2, 2]),
        "dataset_group_count": 10,
        "dataset_video_count": 30,
    }


def _config(mode="pmgr"):
    return {
        "engine": {"video_encoder_microbatch": 2, "text_encoder_microbatch": 2,
                   "score_video_block": 2, "score_text_block": 2},
        "scoring": {"dual_mix": 0.5, "inner_temperature": 0.07,
                    "normalize_eps": 1e-6, "mask_policy": "valid_tokens_only"},
        "loss": {"mode": mode, "rank_mix": 0.25, "rank_eta": 0.1},
    }


def _direct_chunked(model, batch, config):
    videos, video_valid = [], []
    video_microbatch = config["engine"]["video_encoder_microbatch"]
    for start in range(0, len(batch["video_features"]), video_microbatch):
        hidden, valid = model.encode_video(
            batch["video_features"][start : start + video_microbatch],
            batch["video_padding_mask"][start : start + video_microbatch],
        )
        videos.append(hidden)
        video_valid.append(valid)
    text_branches = []
    for prefix in ("", "aug_"):
        values, masks = [], []
        text_microbatch = config["engine"]["text_encoder_microbatch"]
        for start in range(0, len(batch[f"{prefix}input_ids"]), text_microbatch):
            hidden, valid = model.encode_text(
                batch[f"{prefix}input_ids"][start : start + text_microbatch],
                batch[f"{prefix}segment_ids"][start : start + text_microbatch],
                batch[f"{prefix}input_mask"][start : start + text_microbatch],
            )
            values.append(hidden)
            masks.append(valid)
        text_branches.append((torch.cat(values), torch.cat(masks)))
    encoded = EncodedPMGRBatch(torch.cat(videos), torch.cat(video_valid), *text_branches[0], *text_branches[1])
    q, a, b = mixed_pair_scores(
        encoded.video_hidden, encoded.text_hidden, encoded.aug_hidden,
        encoded.video_valid, encoded.text_valid, encoded.aug_valid,
    )
    loss = objective_loss(
        config["loss"]["mode"], q, batch["video_to_group"], model.logit_scale,
        batch["dataset_group_count"], batch["dataset_video_count"], a=a, b=b,
        rank_mix=config["loss"]["rank_mix"], rank_eta=config["loss"]["rank_eta"],
    )["loss"]
    loss.backward()
    return loss.detach(), q.detach()


@pytest.mark.parametrize("mode", ["pmgr", "group_ce"])
def test_direct_and_two_level_parameter_gradient_and_update_match(mode):
    initial = TinyRetriever()
    direct = copy.deepcopy(initial)
    replay = copy.deepcopy(initial)
    batch = _batch()
    config = _config(mode)
    rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    direct_loss, direct_q = _direct_chunked(direct, batch, config)
    direct_gradients = {name: parameter.grad.clone() for name, parameter in direct.named_parameters()}
    torch.set_rng_state(rng)
    result = two_level_backward(replay, batch, config)
    assert float(result.loss) == pytest.approx(float(direct_loss), abs=1e-12)
    torch.testing.assert_close(result.q, direct_q, atol=1e-12, rtol=1e-12)
    for name, parameter in replay.named_parameters():
        torch.testing.assert_close(parameter.grad, direct_gradients[name], atol=1e-11, rtol=1e-11)
    direct_optimizer = torch.optim.SGD(direct.parameters(), lr=0.03)
    replay_optimizer = torch.optim.SGD(replay.parameters(), lr=0.03)
    direct_optimizer.step()
    replay_optimizer.step()
    for left, right in zip(direct.parameters(), replay.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_shared_original_and_augmented_text_contributions_are_both_present():
    model = TinyRetriever()
    batch = _batch()
    torch.manual_seed(99)
    result = two_level_backward(model, batch, _config())
    assert result.text_gradient.norm() > 0
    assert result.aug_gradient.norm() > 0
    assert model.shared.weight.grad is not None and model.shared.weight.grad.norm() > 0


def test_all_pair_blocks_accumulate_into_text_and_video_buffers():
    model = TinyRetriever()
    batch = _batch()
    torch.manual_seed(4)
    result = two_level_backward(model, batch, _config())
    assert result.block_replays == 6
    assert torch.count_nonzero(result.video_gradient) > 0
    assert torch.count_nonzero(result.text_gradient) > 0
    assert torch.count_nonzero(result.aug_gradient) > 0


def test_legacy_c0_direct_and_cache_parity_uses_both_channel_payloads():
    batch = _batch()
    keep = torch.tensor([0, 2, 3])
    batch["video_features"] = batch["video_features"].index_select(0, keep)
    batch["video_padding_mask"] = batch["video_padding_mask"].index_select(0, keep)
    batch["video_to_group"] = torch.arange(3)
    config = _config("legacy_cico")
    config["loss"]["rank_mix"] = 0.0
    initial = TinyRetriever()
    direct = copy.deepcopy(initial)
    replay = copy.deepcopy(initial)
    rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    direct_loss, _ = _direct_chunked(direct, batch, config)
    torch.set_rng_state(rng)
    result = two_level_backward(replay, batch, config)
    assert float(result.loss) == pytest.approx(float(direct_loss), abs=1e-12)
    for (name, left), (_, right) in zip(
        direct.named_parameters(), replay.named_parameters(), strict=True
    ):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-11, rtol=1e-11, msg=name)
