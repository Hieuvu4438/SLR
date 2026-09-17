"""Source-level loss identity fixtures; no dataset, checkpoint or training.

Run from the repository root. This does not establish realizability of arbitrary
logit matrices by CiCo's coupled token-scoring channels or retrieval harm.
"""
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def gap(a, b):
    """Mean negative log Bhattacharyya coefficient, computed in log space."""
    return -torch.logsumexp((a.log_softmax(-1) + b.log_softmax(-1)) / 2, -1).mean()


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    source = root / 'third_party/SLRT/CiCo/CLCL/modules/until_module.py'
    raw = source.read_bytes()
    tree = ast.parse(raw)
    node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CrossEn']
    scope = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    loss = scope['CrossEn']()
    # Fixed fixtures declared before execution; no selected random realization.
    base = 4 * torch.eye(4, dtype=torch.float64)
    offset = torch.tensor([-2., -1., 1., 2.], dtype=torch.float64)[:, None]
    contrast = torch.tensor([[0., 6., -2., -4.], [-6., 0., 4., 2.],
                             [2., -4., 0., 6.], [4., -2., -6., 0.]], dtype=torch.float64)
    fixtures = [('identical', base, base), ('row_offset', base, base + offset),
                ('complementary', base + contrast, base - contrast),
                ('large_offset', base + 1000., base - 1000.)]
    records = []
    for name, aa, bb in fixtures:
        a, b = aa.clone().requires_grad_(), bb.clone().requires_grad_()
        m = (a + b) / 2
        separate = (loss(a) + loss(a.T) + loss(b) + loss(b.T)) / 4
        mixed = (loss(m) + loss(m.T)) / 2
        j = (gap(a, b) + gap(a.T, b.T)) / 2
        direct = ((a.logsumexp(-1) - a.diag()).mean() +
                  (a.T.logsumexp(-1) - a.diag()).mean() +
                  (b.logsumexp(-1) - b.diag()).mean() +
                  (b.T.logsumexp(-1) - b.diag()).mean()) / 4
        g1 = torch.autograd.grad(separate, (a, b), retain_graph=True)
        g2 = torch.autograd.grad(mixed + j, (a, b))
        errors = {'loss_identity': float((separate - mixed - j).detach().abs()),
                  'source_vs_logsumexp': float((separate - direct).detach().abs()),
                  'gradient_identity': max(float((x-y).abs().max()) for x,y in zip(g1,g2))}
        assert max(errors.values()) <= 1e-10, errors
        assert j.detach() >= -1e-12
        if name in ('identical', 'large_offset'):
            assert j.detach().abs() < 1e-12
        if name == 'row_offset':
            assert gap(a, b).detach().abs() < 1e-12
            assert gap(a.T, b.T).detach() > 0
        records.append({'fixture': name, 'separate_loss': float(separate.detach()),
                        'mixed_loss': float(mixed.detach()), 'agreement_gap': float(j.detach()),
                        'errors': errors})
    print(json.dumps({'status': 'completed', 'scope': 'four fixed synthetic logit fixtures only',
                      'dtype': 'float64', 'torch_version': torch.__version__,
                      'source_sha256': hashlib.sha256(raw).hexdigest(),
                      'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'data_or_checkpoint_access': False, 'optimizer_updates': 0,
                      'method_go': False, 'records': records}, indent=2))


if __name__ == '__main__':
    main()
