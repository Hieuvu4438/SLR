"""TRAIN-only CiCo LayerNorm radial reconstruction; no model or scoring run."""
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .common import ART, ROOT, dump, sha

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


def inverse_coefficients(projection, gamma, beta):
    p, g, bias = projection.double(), gamma.double(), beta.double()
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError('square projection required for this inverse')
    if g.shape != (len(p),) or bias.shape != g.shape or (g == 0).any():
        raise ValueError('aligned nonzero LayerNorm gains required')
    b = (bias / g).sum()
    if b == 0:
        raise ValueError('zero affine offset: this inverse is uninformative')
    a = torch.linalg.solve(p, 1 / g)
    return a, b


def fixed_controls():
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(32, 16, dtype=torch.float64, generator=gen)
    p = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64, generator=gen)).Q
    gamma = torch.linspace(.5, 1.5, 16, dtype=torch.float64)
    beta = .2 * gamma
    z = F.layer_norm(x, (16,), gamma, beta, 1e-5) @ p
    a, b = inverse_coefficients(p, gamma, beta)
    radius = z.norm(dim=-1)
    reconstructed = b / (F.normalize(z, dim=-1) @ a)
    error = float(((reconstructed - radius) / radius).abs().max())
    assert error < 1e-10
    refused = []
    for name, matrix, bias in [('zero_bias', p, torch.zeros_like(beta)),
                               ('rectangular', p[:, :8], beta)]:
        try:
            inverse_coefficients(matrix, gamma, bias)
        except ValueError:
            refused.append(name)
        else:
            raise AssertionError(name)
    return {'max_relative_error': error, 'unsupported_cases_refused': refused}


def summary(errors, signed, denominators, norms):
    finite = torch.isfinite(errors) & torch.isfinite(signed)
    e = errors[finite]
    quantiles = None if not len(e) else {
        'median': float(e.quantile(.5)), 'p95': float(e.quantile(.95)),
        'max': float(e.max())}
    return {
        'n': len(errors), 'nonfinite_n': int((~finite).sum()),
        'nonpositive_reconstructed_n': int((signed <= 0).sum()),
        'zero_denominator_n': int((denominators == 0).sum()),
        'minimum_abs_denominator': float(denominators.abs().min()),
        'radius_min': float(norms.min()), 'radius_max': float(norms.max()),
        'relative_error': quantiles,
        'fractions_all_slots_within_relative_error': {
            str(t): float((finite & (errors <= t)).double().mean())
            for t in (.001, .01, .05)},
        'approximate_recovery_screen': bool(
            finite.all() and (signed > 0).all() and quantiles['p95'] <= .01),
    }


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    start = time.time()
    dest = OUT / 'CICO-RADIAL-CONTRACT.json'
    if dest.exists():
        raise FileExistsError(dest)
    controls = fixed_controls()
    checkpoint = ROOT / 'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
    cache_path = ART / 'frozen_train.pt'
    manifest = ROOT / 'artifacts/manifests/ph_train.jsonl'
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['model']
    assert cache['checkpoint_sha256'] == sha(checkpoint)
    assert cache['manifest_sha256'] == sha(manifest)
    assert len(cache['ids']) == 7096
    p = state['core.clip.text_projection'].double()
    gamma = state['core.clip.ln_final.weight'].double()
    beta = state['core.clip.ln_final.bias'].double()
    a, b = inverse_coefficients(p, gamma, beta)
    singular = torch.linalg.svdvals(p)
    mask = cache['text_mask'].flatten() == 1
    records = {}
    for precision in ('float64_direction', 'float32_direction'):
        errors, signed, denominators, norms = [], [], [], []
        for i in range(0, len(cache['ids']), 128):
            z = cache['text_tokens'][i:i+128].reshape(-1, p.shape[1]).double()
            radius = z.norm(dim=-1)
            assert (radius > 0).all()
            direction = F.normalize(z, dim=-1)
            if precision == 'float32_direction':
                direction = direction.float().double()
            denominator = direction @ a
            reconstructed = b / denominator
            errors.append(((reconstructed - radius) / radius).abs())
            signed.append(reconstructed)
            denominators.append(denominator)
            norms.append(radius)
        values = [torch.cat(v) for v in (errors, signed, denominators, norms)]
        records[precision] = {
            'all_slots': summary(*values),
            'mask_valid': summary(*(v[mask] for v in values)),
        }
    sources = [Path(__file__), OUT / 'CICO_radial_contract_protocol.md',
               ROOT / 'third_party/SLRT/CiCo/CLCL/modules/module_clip.py',
               ROOT / 'third_party/SLRT/CiCo/CLCL/modules/modeling.py',
               ROOT / 'methods/information_probe/cache_baseline.py',
               ROOT / 'shared/slr_common/upstream/cico_bridge.py']
    report = {
        'status': 'completed', 'scope': 'fixed synthetic controls and existing TRAIN text cache',
        'torch_version': torch.__version__, 'controls': controls,
        'checkpoint_sha256': cache['checkpoint_sha256'],
        'cache_sha256': sha(cache_path), 'manifest_sha256': cache['manifest_sha256'],
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
        'text_projection_shape': list(p.shape),
        'video_projection_shape': list(state['core.clip.visual.proj'].shape),
        'text_projection_singular_min': float(singular.min()),
        'text_projection_condition': float(singular.max() / singular.min()),
        'gamma_min_abs': float(gamma.abs().min()), 'affine_b': float(b),
        'coefficient_norm': float(a.norm()),
        'relative_solve_residual': float((p @ a - 1/gamma).norm() / (1/gamma).norm()),
        'records': records, 'optimizer_updates': 0, 'encoder_executions': 0,
        'dev_test_cache_access': False, 'method_go': False,
        'wall_seconds': time.time() - start,
    }
    dump(dest, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
