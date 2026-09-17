"""Unchanged CMCM Gaussian module on matched distributions; synthetic CPU only."""
import ast
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'third_party/CMCM/modules/CCG_Module.py'


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    raw = SOURCE.read_bytes()
    tree = ast.parse(raw)
    names = {'CausalAttention', 'GaussianParameterization', 'GaussianAlignmentModule'}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == 3
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    model = namespace['GaussianAlignmentModule'](embed_dim=8, num_heads=2, align_dim=3).double()
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.zeros(1, 1024, 8, dtype=torch.float64)
    results = []
    for bias in [0.0, -10.0, -18.0, -25.0, -40.0, -80.0]:
        with torch.no_grad():
            model.video_gaussian.sigma_proj[0].bias.fill_(bias)
            model.cross_modal_gaussian.sigma_proj[0].bias.fill_(bias)
        model.zero_grad(set_to_none=True)
        loss = model(x, x)
        loss.backward()
        v = math.log1p(math.exp(bias))
        epsilon = 1e-8
        reference = 0.5 * (math.log(v / (v + epsilon)) + v / (v + epsilon) - 1)
        derivative = 0.5 * (epsilon / (v * (v + epsilon)) + epsilon / (v + epsilon)**2) / (1 + math.exp(-bias))
        video_derivative = model.video_gaussian.sigma_proj[0].bias.grad.sum().item()
        text_derivative = model.cross_modal_gaussian.sigma_proj[0].bias.grad.sum().item()
        actual_derivative = video_derivative + text_derivative
        p = torch.distributions.Normal(torch.zeros(3, dtype=torch.float64), math.sqrt(v))
        shifted = torch.distributions.Normal(torch.zeros(3, dtype=torch.float64), math.sqrt(v + epsilon))
        exact_kl = torch.distributions.kl_divergence(p, p).mean().item()
        shifted_kl = torch.distributions.kl_divergence(shifted, shifted).mean().item()
        assert math.isfinite(loss.item()) and math.isfinite(actual_derivative)
        assert loss.item() < 0 and actual_derivative > 0
        assert abs(loss.item() - reference) < 1e-10
        assert abs(actual_derivative - derivative) < 1e-10
        assert exact_kl == shifted_kl == 0
        results.append({
            'shared_bias': bias, 'variance': v, 'source_loss': loss.item(),
            'scalar_reference': reference, 'loss_absolute_error': abs(loss.item() - reference),
            'video_bias_directional_derivative': video_derivative,
            'cross_modal_bias_directional_derivative': text_derivative,
            'shared_bias_directional_derivative': actual_derivative,
            'analytic_directional_derivative': derivative,
            'gradient_absolute_error': abs(actual_derivative - derivative),
            'exact_kl': exact_kl, 'consistent_shift_kl': shifted_kl,
        })
    assert all(a['source_loss'] > b['source_loss'] for a, b in zip(results, results[1:]))
    result = {
        'scope': 'Synthetic matched diagonal Gaussians; no data, trained weights or optimizer',
        'torch_version': torch.__version__, 'dtype': 'float64', 'device': 'cpu',
        'shape': [1, 1024, 8], 'heads': 2, 'alignment_dimension': 3,
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fixtures': results, 'all_assertions_pass': True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
