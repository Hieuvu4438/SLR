"""Scoped native-operator tests; no dataset/GPU/baseline campaign."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/SEDS')]
from modules.modeling_gcn import ConvTemporalGraphical
from methods.seds_adaptation.adaptive_graph import attach_graph_delta
from methods.seds_adaptation.masked_pose import masked_learning_rates


class AdaptiveGraphTest(unittest.TestCase):
    def test_native_parity_gradient_reload_and_no_cumulative_graph(self):
        torch.manual_seed(42)
        graph = torch.eye(7).repeat(3, 1, 1)
        anchor = graph.clone()
        native = ConvTemporalGraphical(2, 8, 3).eval()
        adapted = copy.deepcopy(native)
        attach_graph_delta(adapted, graph)
        x = torch.randn(2, 2, 1, 7)
        self.assertTrue(torch.equal(native(x, graph)[0], adapted(x, graph)[0]))
        output, returned = adapted(x, graph)
        self.assertIs(returned, graph)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(adapted.graph_delta.grad).all())
        self.assertGreater(float(adapted.graph_delta.grad.abs().sum()), 0)
        with torch.no_grad():
            adapted.graph_delta.add_(-.01*adapted.graph_delta.grad)
        self.assertFalse(torch.equal(native(x, graph)[0], adapted(x, graph)[0]))
        self.assertTrue(torch.equal(graph, anchor))
        reloaded = copy.deepcopy(native)
        attach_graph_delta(reloaded, graph)
        reloaded.load_state_dict(adapted.state_dict(), strict=True)
        self.assertTrue(torch.equal(reloaded(x, graph)[0], adapted(x, graph)[0]))
        with self.assertRaises(ValueError):
            attach_graph_delta(adapted, graph)

    def test_lr_partition(self):
        model = nn.Module()
        model.signbert = nn.Module()
        model.signbert.embed = ConvTemporalGraphical(2, 8, 3)
        attach_graph_delta(model.signbert.embed, torch.eye(7).repeat(3, 1, 1))
        model.fusion = nn.Linear(8, 2)
        opt = torch.optim.Adam(model.parameters())
        groups = masked_learning_rates(opt, model, control=True, adaptive_graph=True)
        self.assertEqual({g['adaptation_group']: g['lr'] for g in groups},
                         dict(encoder=1e-6, fusion=1e-5, graph=1e-4))
        params = [p for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        slow_opt = torch.optim.Adam(model.parameters())
        slow = masked_learning_rates(slow_opt, model, control=True, adaptive_graph=True, graph_lr=1e-5)
        self.assertEqual({g['adaptation_group']: g['lr'] for g in slow},
                         dict(encoder=1e-6, fusion=1e-5, graph=1e-5))
        # Only graph updates should differ by exactly tenfold for the same gradient.
        fast_graph = next(g for g in opt.param_groups if g['adaptation_group']=='graph')
        slow_graph = next(g for g in slow_opt.param_groups if g['adaptation_group']=='graph')
        self.assertAlmostEqual(fast_graph['lr']/slow_graph['lr'],10.)


if __name__ == '__main__':
    unittest.main()
