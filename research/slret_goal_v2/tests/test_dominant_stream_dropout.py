import copy
import unittest

import torch
from torch import nn

from methods.seds_adaptation.dominant_stream_dropout import (
    AnnealedRGBDropout, attach_annealed_rgb_dropout)


class ToyFusion(nn.Module):
    def forward(self, pose, rgb, mask):
        return pose + 2 * rgb


class DominantStreamDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.rgb = torch.randn(8, 4, 6)

    def test_linear_schedule_reaches_zero(self):
        module = AnnealedRGBDropout(.2)
        module.set_step(1, 11)
        self.assertAlmostEqual(module.probability, .2)
        module.set_step(6, 11)
        self.assertAlmostEqual(module.probability, .1)
        module.set_step(11, 11)
        self.assertEqual(module.probability, 0)

    def test_eval_is_exact_identity_and_checkpoint_schema_unchanged(self):
        model = nn.Module(); model.fusion = ToyFusion()
        pose = torch.randn_like(self.rgb); mask = torch.zeros(8, 4, dtype=torch.bool)
        expected = model.fusion(pose, self.rgb, mask)
        keys = set(model.state_dict())
        module = attach_annealed_rgb_dropout(model, .2, 7)
        model.eval()
        actual = model.fusion(pose, self.rgb, mask)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(module.eval_identity_passed)
        self.assertEqual(set(model.state_dict()), keys)

    def test_training_drops_only_rgb_without_mutating_input(self):
        module = AnnealedRGBDropout(1., 7).train()
        module.set_step(1, 2)
        original = self.rgb.clone()
        result = module(self.rgb)
        self.assertEqual(torch.count_nonzero(result), 0)
        self.assertTrue(torch.equal(self.rgb, original))
        self.assertEqual(module.stats['rgb_dropped_samples'], len(self.rgb))

    def test_terminal_training_step_uses_complete_input(self):
        module = AnnealedRGBDropout(.2, 7).train()
        module.set_step(5, 5)
        result = module(self.rgb)
        self.assertTrue(torch.equal(result, self.rgb))
        self.assertEqual(module.stats['rgb_dropped_samples'], 0)

    def test_private_generator_preserves_global_rng(self):
        module = AnnealedRGBDropout(.2, 7).train(); module.set_step(1, 5)
        before = torch.random.get_rng_state().clone()
        module(self.rgb)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_private_seed_is_reproducible(self):
        a = AnnealedRGBDropout(.2, 31).train(); b = copy.deepcopy(a).train()
        a.set_step(1, 5); b.set_step(1, 5)
        self.assertTrue(torch.equal(a(self.rgb), b(self.rgb)))

    def test_invalid_contracts_rejected(self):
        with self.assertRaises(ValueError): AnnealedRGBDropout(0)
        module = AnnealedRGBDropout(.2)
        with self.assertRaises(ValueError): module.set_step(0, 5)
        with self.assertRaises(ValueError): module(torch.randn(2, 3))


if __name__ == '__main__':
    unittest.main()
