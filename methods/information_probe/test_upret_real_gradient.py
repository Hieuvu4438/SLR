"""Isolated arithmetic tests without importing a model or opening assets."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def helpers():
    tree = ast.parse(Path(__file__).with_name('upret_real_gradient.py').read_text())
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in {'objective', 'score_energy', 'gradient_group'}]
    scope = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<isolated-helpers>', 'exec'), scope)
    return scope


def test_native_balanced_objective_and_ot_scale_placement():
    h = helpers()
    core = SimpleNamespace(clip=SimpleNamespace(logit_scale=torch.tensor(2.).log()),
                           dual_mix=.5, ot_weight=1.)
    v = torch.tensor([[.2, .1], [.7, -.3]], dtype=torch.float64, requires_grad=True)
    t = v.T + .1
    ot = torch.tensor([[.1, .5], [.2, -.1]], dtype=torch.float64)
    lv, lt = 2*v + ot, 2*t + ot
    expected = -.25*sum(x.log_softmax(-1).diagonal().mean() for x in (lv, lv.T, lt, lt.T))
    actual = h['objective'](core, v, t, ot)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, v, retain_graph=True)[0],
                       torch.autograd.grad(expected, v)[0])


def test_amplitude_calibration_and_parameter_partition():
    h = helpers()
    x = torch.tensor([[1., 2.], [3., 5.]], dtype=torch.float64)
    c = (h['score_energy'](x)/h['score_energy'](4*x+7)).sqrt()
    assert c == .25
    assert h['score_energy'](c*(4*x+7)) == h['score_energy'](x)
    assert h['gradient_group']('clip.visual.proj') == 'visual_encoder'
    assert h['gradient_group']('clip.text_projection') == 'text_encoder'
    assert h['gradient_group']('clip.logit_scale') == 'score_heads'
