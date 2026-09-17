"""Feasible final-layer visual radial collisions, not encoder/data collisions."""
import ast
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ART, ROOT, dump, sha

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    output = OUT / 'CICO-VISUAL-RADIAL.json'
    if output.exists():
        raise FileExistsError(output)
    cp = ROOT / 'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
    cache_path = ART / 'frozen_train.pt'
    cache = torch.load(cache_path, weights_only=True, map_location='cpu', mmap=True)
    state = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)['model']
    assert cache['checkpoint_sha256'] == sha(cp)
    assert cache['manifest_sha256'] == sha(ROOT / 'artifacts/manifests/ph_train.jsonl')
    assert len(cache['ids']) == 7096
    assert (cache['video_mask'][:256, 1] == 0).all()
    p = state['core.clip.visual.proj'].double()
    g = state['core.clip.visual.ln_post.weight'].double()
    beta = state['core.clip.visual.ln_post.bias'].double()
    dim = len(g)
    matrix = torch.cat(((g[:, None] * p).T, torch.ones(1, dim).double() / math.sqrt(dim)))
    singular = torch.linalg.svdvals(matrix)
    rank = int((singular > singular.max() * 1e-12).sum())
    assert rank == matrix.shape[0], 'full row-rank prerequisite failed'
    inv = torch.linalg.pinv(matrix, rtol=1e-12)
    z = cache['video_tokens'][:256, 1].double()
    target_q = 1 - 1e-5
    target_sqnorm = dim * target_q
    eps = 1e-5
    summaries, minima = {}, {}
    for scale in (.9, 1.1):
        targets = scale * z
        rhs = torch.cat((targets - beta @ p, torch.zeros(256, 1)), -1)
        u = rhs @ inv.T
        square_norms = u.square().sum(-1)
        summaries[str(scale)] = {
            'n': len(z), 'feasible_n': int((square_norms < target_sqnorm).sum()),
            'minimum_required_norm_squared_over_D_max': float(square_norms.max() / dim),
            'minimum_required_norm_squared_over_D_median': float(square_norms.median() / dim),
            'max_constraint_residual': float((u @ matrix.T - rhs).abs().max()),
        }
        minima[scale] = u
    constructed = {'row': 0, 'performed': False}
    if all(minima[s][0].square().sum() < target_sqnorm for s in (.9, 1.1)):
        gen = torch.Generator().manual_seed(42)
        direction = torch.randn(dim, dtype=torch.float64, generator=gen)
        direction -= inv @ (matrix @ direction)
        direction /= direction.norm()
        inputs, actual = [], []
        for scale in (.9, 1.1):
            base = minima[scale][0]
            u = base + (target_sqnorm - base.square().sum()).sqrt() * direction
            h = u * math.sqrt(eps / (1 - target_q))
            inputs.append(h)
            actual.append(F.layer_norm(h, (dim,), g, beta, eps) @ p)
        inputs, actual = torch.stack(inputs), torch.stack(actual)
        targets = torch.stack((.9 * z[0], 1.1 * z[0]))
        relative = (actual - targets).norm(dim=-1) / targets.norm(dim=-1)
        normed = F.normalize(actual, dim=-1)
        direction_error = float((normed[0] - normed[1]).abs().max())
        ratio = float(actual[1].norm() / actual[0].norm())
        assert relative.max() < 1e-9 and direction_error < 1e-9
        assert abs(ratio - 1.1/.9) < 1e-9
        source = ROOT / 'third_party/SLRT/CiCo/CLCL/modules/module_clip.py'
        node, = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.ClassDef) and n.name == 'LayerNorm']
        scope = {'torch': torch, 'nn': nn}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        layer = scope['LayerNorm'](dim, eps=eps)
        layer.weight.copy_(g.float())
        layer.bias.copy_(beta.float())
        native = (layer(inputs.half()) @ p.half()).double()
        native_d = F.normalize(native, dim=-1)
        constructed = {
            'row': 0, 'performed': True,
            'fp64_relative_target_error_max': float(relative.max()),
            'fp64_normalized_direction_max_delta': direction_error,
            'fp64_radius_ratio': ratio,
            'input_mean_abs_max': float(inputs.mean(-1).abs().max()),
            'input_variances': inputs.var(-1, unbiased=False).tolist(),
            'input_distance': float((inputs[0] - inputs[1]).norm()),
            'source_cpu_fp16_relative_target_error_max': float(
                ((native - targets).norm(dim=-1) / targets.norm(dim=-1)).max()),
            'source_cpu_fp16_direction_max_delta': float((native_d[0]-native_d[1]).abs().max()),
            'source_cpu_fp16_radius_ratio': float(native[1].norm()/native[0].norm()),
        }
    report = {
        'status': 'completed', 'matrix_shape': list(matrix.shape), 'matrix_rank': rank,
        'matrix_condition': float(singular.max()/singular.min()),
        'scales': summaries, 'construction': constructed,
        'checkpoint_sha256': cache['checkpoint_sha256'], 'cache_sha256': sha(cache_path),
        'manifest_sha256': cache['manifest_sha256'], 'code_sha256': sha(__file__),
        'protocol_sha256': sha(OUT / 'CICO_visual_radial_protocol.md'),
        'source_sha256': sha(ROOT / 'third_party/SLRT/CiCo/CLCL/modules/module_clip.py'),
        'torch_version': torch.__version__, 'method_go': False,
        'encoder_executions': 0, 'optimizer_updates': 0, 'dev_test_access': False,
        'scope': 'Local final-layer function class, NOT actual video-pair collisions.',
    }
    dump(output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
