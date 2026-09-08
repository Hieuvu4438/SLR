from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from dive.adapters import (
    BaselineAdapter,
    SedsAdapter,
    SedsAdapterError,
    SedsLocalPoseEncoder,
    SedsReproductionError,
    SedsTextBatch,
    SedsVideoBatch,
    normalize_seds_text_mask,
    normalize_seds_video_mask,
    seds_prelogit_fusion_scores,
    load_seds_reproduction,
)
from dive.config import load_config
from dive.data.text_units import MappedTextUnit, TextUnit
from dive.models.evidence import EvidenceEncoder
from dive.training.optimizer import build_evidence_optimizer
from dive.training.warmup import WarmupBatch, run_warmup_step


ROOT = Path(__file__).resolve().parents[3]
SEDS_ROOT = ROOT / "third_party" / "SEDS"
REPRODUCTION_CONFIG = ROOT / "methods" / "dive" / "configs" / "seds_how2sign_reproduction.yaml"


class FakeSignbert(nn.Module):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(dimension))

    def gcn_emb(self, pose):
        return {"feat": self.projection(pose["body"])}

    def sign_conv(self, features):
        return features


class FakeFusion(nn.Module):
    def forward(self, pose, rgb, mask):
        del mask
        return pose + rgb


class FakeClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))


class FakeSeds(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.signbert = FakeSignbert()
        self.fusion = FakeFusion()
        self.clip = FakeClip()
        self.task_config = SimpleNamespace(slide_windows=16, feature_len=64)
        self.fusion_type = "gloss_atten"
        self.sim_header = "Filip"
        self.dual_mix = 0.5

    def get_visual_output(self, right, left, body, shaped=True, get_hidden=True):
        del right, left, shaped, get_hidden
        rgb = body["rgb"].squeeze(-1).transpose(1, 2)
        pose = body["pose"][:, : rgb.shape[1]]
        rgb_hidden = torch.cat((rgb.mean(dim=1, keepdim=True), rgb), dim=1)
        pose_hidden = torch.cat((pose.mean(dim=1, keepdim=True), pose), dim=1)
        return body["mask"], pose_hidden, rgb_hidden

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, **kwargs):
        del token_type_ids, kwargs
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        return attention_mask, hidden

    def get_similarity_logits(
        self,
        sequence_output,
        visual_hidden_pose,
        visual_hidden_rgb,
        attention_mask,
        video_mask,
        **kwargs,
    ):
        del kwargs
        mixed, i2t, t2i = seds_prelogit_fusion_scores(
            self.fusion(visual_hidden_pose, visual_hidden_rgb, video_mask),
            sequence_output,
            video_mask == 0,
            attention_mask == 1,
            dual_mix=self.dual_mix,
        )
        del mixed
        scale = self.clip.logit_scale.exp()
        return i2t * scale, t2i * scale, None, None, None, None, 0.0, 0.0, None, None


def _video_batch() -> SedsVideoBatch:
    rgb_local = torch.tensor(
        [
            [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.5, 0.0]],
            [[0.5, 0.5, 1.0, 0.0], [0.2, 0.8, 0.0, 1.0]],
        ]
    )
    pose_frames = rgb_local[:, [0] * 16 + [1] * 16]
    return SedsVideoBatch(
        sample_ids=("v0", "v1"),
        right_pose=pose_frames.clone(),
        left_pose=pose_frames.clone(),
        body_pose=pose_frames,
        clip_starts=torch.tensor([[0, 16], [0, 16]]),
        legacy_video_mask=torch.tensor([[0, 0, 0], [0, 0, 0]]),
        rgb_features=rgb_local.transpose(1, 2).unsqueeze(-1),
        grid_id="canonical",
        raw_frame_counts=(32, 32),
        frames_per_second=(25.0, 25.0),
    )


def _text_batch() -> SedsTextBatch:
    return SedsTextBatch(
        text_ids=("t0", "t1"),
        input_ids=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        token_type_ids=torch.zeros(2, 3, dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
    )


def test_pure_seds_prelogit_matches_unpadded_upstream_formula_and_orientation():
    generator = torch.Generator().manual_seed(211)
    video = torch.randn(2, 3, 5, generator=generator)
    text = torch.randn(4, 2, 5, generator=generator)
    vm = torch.ones(2, 3, dtype=torch.bool)
    tm = torch.ones(4, 2, dtype=torch.bool)
    score, i2t, t2i = seds_prelogit_fusion_scores(video, text, vm, tm, dual_mix=0.3)
    normalized_video = torch.nn.functional.normalize(video, dim=-1)
    normalized_text = torch.nn.functional.normalize(text, dim=-1)
    similarities = torch.einsum("vnd,tmd->vtnm", normalized_video, normalized_text)
    upstream_i2t = (similarities * torch.softmax(similarities / 0.07, dim=3)).sum(3).mean(2)
    upstream_t2i = (similarities * torch.softmax(similarities / 0.07, dim=2)).sum(2).mean(2)
    assert score.shape == i2t.shape == t2i.shape == (2, 4)
    torch.testing.assert_close(i2t, upstream_i2t)
    torch.testing.assert_close(t2i, upstream_t2i)
    torch.testing.assert_close(score, 0.3 * upstream_i2t + 0.7 * upstream_t2i)


def test_seds_score_masks_padding_before_softmax():
    video = torch.nn.functional.normalize(torch.randn(2, 2, 4), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(3, 2, 4), dim=-1)
    base = seds_prelogit_fusion_scores(
        video,
        text,
        torch.ones(2, 2, dtype=torch.bool),
        torch.ones(3, 2, dtype=torch.bool),
    )[0]
    padded_video = torch.cat((video, torch.full((2, 1, 4), 1000.0)), dim=1)
    padded_text = torch.cat((text, torch.full((3, 1, 4), -1000.0)), dim=1)
    padded = seds_prelogit_fusion_scores(
        padded_video,
        padded_text,
        torch.tensor([[True, True, False], [True, True, False]]),
        torch.tensor([[True, True, False]] * 3),
    )[0]
    torch.testing.assert_close(padded, base)


def test_seds_masks_remove_only_explicit_cls_for_local_features():
    legacy = torch.tensor([[0, 0, 1], [0, 1, 1]])
    assert torch.equal(
        normalize_seds_video_mask(legacy, local_length=2),
        torch.tensor([[True, False], [False, False]]),
    )
    assert torch.equal(
        normalize_seds_text_mask(torch.tensor([[1, 1, 0]]), (1, 3)),
        torch.tensor([[True, True, False]]),
    )
    with pytest.raises(SedsAdapterError, match="exactly one CLS"):
        normalize_seds_video_mask(legacy, local_length=3)


def test_local_pose_clone_uses_gcn_before_fixed_windows_and_sign_conv():
    encoder = SedsLocalPoseEncoder(FakeSignbert(), slide_windows=2)
    body = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    pose = {"right": body, "left": body, "body": body}
    grid = torch.tensor([[[0, 2], [2, 4]]])
    output = encoder(pose, grid)
    torch.testing.assert_close(output, torch.stack((body[:, :2].mean(1), body[:, 2:].mean(1)), dim=1))


def test_evidence_warmup_accepts_synchronized_seds_pose_tree_without_input_gradients():
    encoder = EvidenceEncoder(
        SedsLocalPoseEncoder(FakeSignbert(), slide_windows=2),
        rgb_dim=4,
        pose_dim=4,
        hidden_dim=8,
        output_dim=4,
    )
    body = torch.randn(2, 4, 4)
    pose = {
        "right": body.clone().requires_grad_(True),
        "left": body.clone().requires_grad_(True),
        "body": body.clone().requires_grad_(True),
    }
    rgb = torch.randn(2, 2, 4)
    grid = torch.tensor([[[0, 2], [2, 4]], [[0, 2], [2, 4]]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    with torch.no_grad():
        text = torch.nn.functional.normalize(encoder(pose, rgb, grid, valid).mean(1), dim=-1)
    batch = WarmupBatch(
        pose=pose,
        rgb_local=rgb,
        grid=grid,
        video_mask=valid,
        text_units=text[:, None, :],
        text_mask=torch.ones(2, 1, dtype=torch.bool),
        positives=torch.eye(2, dtype=torch.bool),
        candidates=torch.ones(2, 2, dtype=torch.bool),
    )
    optimizer = build_evidence_optimizer(encoder).optimizer
    loss, _ = run_warmup_step(
        encoder,
        optimizer,
        batch,
        tau_alignment=0.07,
        tau_retrieval=0.07,
        grad_clip_norm=1.0,
    )
    assert loss >= 0
    assert all(value.grad is None for value in pose.values())
    assert encoder.pose_encoder.signbert.projection.weight.grad is not None


def test_adapter_features_prelogit_units_rf_and_checkpoint_provenance(tmp_path):
    model = FakeSeds()
    checkpoint = tmp_path / "seds.pt"
    torch.save(model.state_dict(), checkpoint)
    config = load_config(ROOT / "methods" / "dive" / "configs" / "fixture.yaml")
    config["baseline"]["reproduction_config"] = str(REPRODUCTION_CONFIG)
    config["baseline"]["locked_checkpoint"] = str(checkpoint)
    adapter = SedsAdapter(model, upstream_root=SEDS_ROOT)
    assert isinstance(adapter, BaselineAdapter)
    with pytest.raises(SedsAdapterError, match="CLIP initialization is missing"):
        SedsAdapter.from_official_checkpoint(
            checkpoint,
            config,
            upstream_root=SEDS_ROOT,
        )
    metadata = adapter.load_and_validate(checkpoint, config)
    assert len(metadata["checkpoint_sha256"]) == 64
    assert len(metadata["model_state_sha256"]) == 64
    assert all(not parameter.requires_grad for parameter in model.parameters())

    video = adapter.encode_video_native(_video_batch())
    text = adapter.encode_text_native(_text_batch())
    score = adapter.score_prelogit(video, text)
    assert score.scores.shape == (2, 2)
    assert score.logit_scale == pytest.approx(10.0)
    assert score.diagnostics["score_orientation"] == "video_rows_text_columns"
    parity = adapter.validate_unpadded_native_parity(video, text)
    assert parity["passed"] is True
    assert parity["i2t_max_abs_error"] < 1e-6

    unit = TextUnit(0, 0, 1, "x", "x", "word")
    mappings = (
        (MappedTextUnit(unit, (1,), (1,), True),),
        (MappedTextUnit(unit, (2,), (2,), True),),
    )
    units = adapter.encode_text_units(_text_batch(), mappings)
    assert units.token_features.shape == (2, 1, 4)
    assert bool(units.token_validity.all())
    torch.testing.assert_close(torch.linalg.vector_norm(units.token_features, dim=-1), torch.ones(2, 1))

    rgb = adapter.rgb_local_features(_video_batch(), "canonical")
    assert rgb.streams["rgb_local"].shape == (2, 2, 4)
    rf = adapter.describe_receptive_field(_video_batch(), "canonical")
    assert len(rf) == 4
    assert rf[0]["interval_convention"] == "half_open"
    assert rf[0]["raw_frame_interval"] == [0, 20]
    assert adapter.describe_preprocessing()["padding_fix"] == "mask_before_directional_softmax"


def test_checked_reproduction_config_binds_flags_types_and_pinned_sources(tmp_path):
    reproduction = load_seds_reproduction(REPRODUCTION_CONFIG, upstream_root=SEDS_ROOT)
    assert reproduction.model_arguments["freeze_exfusion"] is False
    assert reproduction.published_eval_arguments["init_model"] == "ckpts/h2s_best_model.bin"
    assert reproduction.published_train_arguments["epochs"] == 200
    assert reproduction.controlled_protocol["checkpoint_selection"] == "independent_dev_only"

    malformed = yaml.safe_load(REPRODUCTION_CONFIG.read_text(encoding="utf-8"))
    malformed["model_arguments"]["feature_len"] = "64"
    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text(yaml.safe_dump(malformed), encoding="utf-8")
    with pytest.raises(SedsReproductionError, match="values/types mismatch"):
        load_seds_reproduction(malformed_path, upstream_root=SEDS_ROOT)
