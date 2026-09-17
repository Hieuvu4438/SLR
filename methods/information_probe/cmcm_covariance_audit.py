"""Committed CMCM covariance pipeline versus exact-forward autodiff control."""
import ast
import hashlib
import json
import subprocess
import time

import torch

from .common import ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'
COMMIT = '5d458719d1da2f082e188cc44705003d919e7e97'


def reference(features):
    x = features.flatten(2)
    centered = x-x.mean(-1, keepdim=True)
    cov = centered @ centered.transpose(-1, -2) / x.shape[-1]
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
    y = cov/(trace+1e-5)
    z = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)[None]
    identity = z
    for _ in range(3):
        correction = .5*(3*identity-z@y)
        y, z = y@correction, correction@z
    matrix = y*trace.sqrt()
    indices = torch.triu_indices(cov.shape[-1], cov.shape[-1])
    return matrix[:, indices[0], indices[1], None]


def finite_difference(fn, x, h):
    result = torch.zeros_like(x)
    for index in range(x.numel()):
        plus, minus = x.clone(), x.clone()
        plus.flatten()[index] += h
        minus.flatten()[index] -= h
        result.flatten()[index] = (fn(plus)-fn(minus))/(2*h)
    return result


def source_classes():
    source = subprocess.check_output(['git', '-C', str(ROOT/'third_party/CMCM'),
                                      'show', COMMIT+':MPNCOV/MPNCOV.py']).decode()
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {'Covpool', 'Sqrtm', 'Triuvec'}]
    assert len(classes) == 3
    scope = {'torch': torch, 'Function': torch.autograd.Function}
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<committed-cmcm>', 'exec'), scope)
    return scope, hashlib.sha256(source.encode()).hexdigest()


def main():
    path = OUT/'CMCM-COVARIANCE_run.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20260917)
    scope, source_sha = source_classes()
    def pipeline(x):
        return scope['Triuvec'].apply(scope['Sqrtm'].apply(scope['Covpool'].apply(x), 3))
    base = torch.randn(1, 3, 2, 2, dtype=torch.float64)
    base -= base.flatten(2).mean(-1)[:, :, None, None]
    base_trace = base.square().sum()/4
    weight = torch.randn(1, 6, 1, dtype=torch.float64)
    entries = []
    for target_trace in (1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1.):
        x = (base*(target_trace/base_trace).sqrt()).requires_grad_()
        y, control = pipeline(x), reference(x)
        grad, = torch.autograd.grad((y*weight).sum(), x)
        control_grad, = torch.autograd.grad((control*weight).sum(), x)
        h = 1e-5*float(x.detach().norm())
        with torch.no_grad():
            fd = finite_difference(lambda z: (pipeline(z)*weight).sum(), x.detach(), h)
            fd_half = finite_difference(lambda z: (pipeline(z)*weight).sum(), x.detach(), h/2)
        denominator = float(control_grad.norm())
        item = {'target_trace': target_trace,
                'features': x.detach().tolist(), 'source_gradient': grad.tolist(),
                'reference_gradient': control_grad.tolist(), 'finite_difference': fd_half.tolist(),
                'forward_max_error': float((y-control).detach().abs().max()),
                'gradient_max_abs_error': float((grad-control_grad).abs().max()),
                'gradient_relative_error': float((grad-control_grad).norm())/denominator,
                'reference_fd_relative_error': float((control_grad-fd_half).norm())/denominator,
                'fd_step_relative_error': float((fd-fd_half).norm())/denominator}
        assert item['forward_max_error'] <= 1e-12
        assert item['reference_fd_relative_error'] <= 1e-5
        entries.append(item)
    zero = torch.zeros_like(base, requires_grad=True)
    zero_output = pipeline(zero)
    zero_grad, = torch.autograd.grad(zero_output.sum(), zero)
    result = {'status': 'completed', 'commit': COMMIT, 'source_sha256': source_sha,
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'CMCM_covariance_protocol.md'),
              'weight': weight.tolist(), 'cases': entries,
              'discrepancy_cases_gt_1e_3': sum(e['gradient_relative_error'] > 1e-3 for e in entries),
              'zero_forward_finite': bool(torch.isfinite(zero_output).all()),
              'zero_backward_finite': bool(torch.isfinite(zero_grad).all()),
              'dataset_loaded': False, 'checkpoint_loaded': False, 'test_loaded': False,
              'method_go': False, 'wall_seconds': time.monotonic()-started}
    dump(path, result)
    print(json.dumps({**{k:v for k,v in result.items() if k not in {'weight','cases'}},
                      'cases': [{k:v for k,v in e.items() if k not in
                                 {'features','source_gradient','reference_gradient','finite_difference'}}
                                for e in entries]}, indent=2))


if __name__ == '__main__':
    main()
