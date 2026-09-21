import types
import unittest

import torch
from torch import nn

from methods.translation_retrieval.bridge import (
    FrozenUniSignDualEncoder,
    masked_mean,
    pose_tokens,
)
from methods.translation_retrieval.cached import (
    FrozenFeatures,
    ProjectionRetrievalModel,
    extract_frozen_features,
    zero_shot_score_matrix,
)


class _Batch(dict):
    def to(self, device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    def __call__(self, strings, **_kwargs):
        lengths = [max(1, min(len(s.split()), 4)) for s in strings]
        ids = torch.zeros(len(strings), max(lengths), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, length in enumerate(lengths):
            ids[index, :length] = torch.arange(1, length + 1)
            mask[index, :length] = 1
        return _Batch(input_ids=ids, attention_mask=mask)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 8)
        self.ffn = nn.Linear(8, 8)

    def forward(self, input_ids=None, inputs_embeds=None, **_kwargs):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        return types.SimpleNamespace(last_hidden_state=self.ffn(x))


class _Donor(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = types.SimpleNamespace(rgb_support=False)
        self.modes = ["body", "left", "right", "face_all"]
        self.proj_linear = nn.ModuleDict({name: nn.Linear(3, 4) for name in self.modes})
        self.gcn_modules = nn.ModuleDict({name: nn.Conv2d(4, 4, 1) for name in self.modes})
        self.fusion_gcn_modules = nn.ModuleDict({name: nn.Conv2d(4, 4, 1) for name in self.modes})
        self.part_para = nn.Parameter(torch.zeros(16))
        self.pose_proj = nn.Linear(16, 8)
        self.mt5_model = nn.Module()
        self.mt5_model.encoder = _Encoder()
        self.mt5_model.config = types.SimpleNamespace(d_model=8)
        self.mt5_tokenizer = _Tokenizer()
        self.lang = "Chinese"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.donor = _Donor()
        self.parts = {
            "body": torch.randn(2, 3, 3, 3),
            "left": torch.randn(2, 3, 4, 3),
            "right": torch.randn(2, 3, 4, 3),
            "face_all": torch.randn(2, 3, 4, 3),
        }
        self.mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    def test_pose_and_text_embeddings_train_only_projections(self):
        bridge = FrozenUniSignDualEncoder(self.donor, projection_dim=6).train()
        video, text = bridge(self.parts, self.mask, ["hello world", "other sentence"])
        self.assertEqual(video.shape, (2, 6))
        self.assertEqual(text.shape, (2, 6))
        self.assertFalse(self.donor.training)
        self.assertTrue(torch.allclose(video.norm(dim=1), torch.ones(2), atol=1e-5))
        loss = bridge.contrastive_loss(video, text)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(bridge.video_proj.weight.grad)
        self.assertIsNotNone(bridge.text_proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.donor.parameters()))

    def test_masked_mean_excludes_padding(self):
        tokens = torch.tensor([[[1.0], [3.0], [100.0]]])
        self.assertEqual(masked_mean(tokens, torch.tensor([[1, 1, 0]])).item(), 2.0)
        with self.assertRaises(ValueError):
            masked_mean(tokens, torch.zeros(1, 3))

    def test_pose_contract_rejects_mismatched_parts(self):
        self.assertEqual(pose_tokens(self.donor, self.parts).shape, (2, 3, 8))
        broken = dict(self.parts)
        broken["left"] = torch.randn(2, 4, 4, 3)
        with self.assertRaises(ValueError):
            pose_tokens(self.donor, broken)
        self.donor.args.rgb_support = True
        with self.assertRaises(ValueError):
            pose_tokens(self.donor, self.parts)

    def test_frozen_cache_and_zero_shot_matrix(self):
        bridge = FrozenUniSignDualEncoder(self.donor, projection_dim=6)
        batch = (["one", "two"], self.parts, self.mask, ["hello world", "other sentence"])
        features = extract_frozen_features(bridge, [batch], torch.device("cpu"))
        self.assertEqual(features.video.shape, (2, 8))
        self.assertEqual(features.text.shape, (2, 8))
        self.assertEqual(zero_shot_score_matrix(features).shape, (2, 2))
        with self.assertRaises(ValueError):
            FrozenFeatures(["one", "one"], features.video, features.text,
                           features.captions).validate()

    def test_projection_head_has_two_sided_gradients(self):
        model = ProjectionRetrievalModel(8, 8, output_dim=4)
        video = torch.randn(3, 8)
        text = torch.randn(3, 8)
        self.assertEqual(model.score_matrix(video, text).shape, (3, 3))
        loss = model.loss(video, text)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.video_proj.weight.grad)
        self.assertIsNotNone(model.text_proj.weight.grad)
        self.assertIsNotNone(model.logit_scale.grad)

    def test_projection_gallery_accepts_rectangular_video_text_counts(self):
        model = ProjectionRetrievalModel(8, 8, output_dim=4)
        video = torch.randn(5, 8)
        text = torch.randn(3, 8)
        self.assertEqual(model.score_gallery(video, text).shape, (5, 3))
        with self.assertRaises(ValueError):
            model.score_matrix(video, text)


if __name__ == "__main__":
    unittest.main()
