"""Posthoc synthetic precision isolation with native CiCo LayerNorm parameters."""
import ast
import json

import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, dump, sha
from .radial_contract import OUT, inverse_coefficients


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    output = OUT / 'CICO-RADIAL-PRECISION.json'
    if output.exists():
        raise FileExistsError(output)
    cp = ROOT / 'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
    state = torch.load(cp, map_location='cpu', weights_only=True, mmap=True)['model']
    p = state['core.clip.text_projection'].double()
    g = state['core.clip.ln_final.weight'].double()
    beta = state['core.clip.ln_final.bias'].double()
    a, b = inverse_coefficients(p, g, beta)
    source = ROOT / 'third_party/SLRT/CiCo/CLCL/modules/module_clip.py'
    tree = ast.parse(source.read_text())
    node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LayerNorm']
    scope = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    layer = scope['LayerNorm'](512)
    layer.weight.copy_(g.float())
    layer.bias.copy_(beta.float())
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(64, 512, dtype=torch.float64, generator=gen)
    rounded = layer(x.half())
    variants = {
        'fp64_reference': F.layer_norm(x, (512,), g, beta, layer.eps) @ p,
        'rounded_layernorm_only': rounded.double() @ p,
        'native_dtype_stages_cpu': (rounded @ p.half()).double(),
    }
    records = {}
    for name, z in variants.items():
        r = z.norm(dim=-1)
        recovered = b / (F.normalize(z, dim=-1) @ a)
        delta = (z @ a - b) / b
        identity_error = ((recovered / r) - 1 / (1 + delta)).abs().max()
        assert identity_error < 1e-10
        errors = ((recovered - r) / r).abs()
        assert torch.isfinite(errors).all()
        if name == 'fp64_reference':
            assert errors.max() < 1e-9
        records[name] = {
            'relative_error_median': float(errors.quantile(.5)),
            'relative_error_p95': float(errors.quantile(.95)),
            'relative_error_max': float(errors.max()),
            'affine_relative_residual_abs_max': float(delta.abs().max()),
            'error_identity_max': float(identity_error),
            'nonfinite_n': int((~torch.isfinite(errors)).sum()),
            'nonpositive_radius_n': int((recovered <= 0).sum()),
        }
    report = {
        'status': 'completed', 'posthoc': True, 'seed': 42, 'synthetic_rows': 64,
        'source_class': 'LayerNorm', 'layernorm_eps': layer.eps,
        'torch_version': torch.__version__, 'checkpoint_sha256': sha(cp),
        'source_sha256': sha(source), 'code_sha256': sha(__file__),
        'parent_result_sha256': sha(OUT / 'CICO-RADIAL-CONTRACT.json'),
        'protocol_sha256': sha(OUT / 'CICO_radial_precision_validation_protocol.md'),
        'records': records, 'dataset_cache_access': False, 'method_go': False,
        'limits': ['Synthetic pre-LayerNorm inputs, not real hidden states.',
                   'CPU dtype emulation, not GPU-kernel parity.',
                   'Original real-cache 1% recovery screen remains failed.'],
    }
    dump(output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
