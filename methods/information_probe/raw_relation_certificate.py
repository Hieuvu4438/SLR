"""AS-C44 structural dependency and synthetic readout-invariance certificate."""
import importlib.util
import json
import math
import os
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from slr_common.features.i3d import ExtractionRecipe
from .common import ROOT, dump, sha
from .raw_readout import RawReadout

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def local_windows(length, kernel, stride):
    padding = max(kernel - (stride if length % stride == 0 else length % stride), 0)
    left = padding // 2
    return [[p for p in range(i*stride-left, i*stride-left+kernel) if 0 <= p < length]
            for i in range(math.ceil(length/stride))]


def set_step(support, kernel, stride):
    return [set().union(*(support[p] for p in window))
            for window in local_windows(len(support), kernel, stride)]


def boolean_step(support, kernel, stride):
    # Independent padding construction and matrix incidence multiplication.
    n = len(support)
    output_n = (n + stride - 1) // stride
    pad = max((output_n - 1)*stride + kernel - n, 0)
    connectivity = np.zeros((output_n, n), dtype=np.int32)
    for target in range(output_n):
        for source in range(n):
            relative = source - (target*stride - pad//2)
            connectivity[target, source] = 0 <= relative < kernel
    return connectivity @ support.astype(np.int32) > 0


def axis_certificate(model, axis, size, pooled_size):
    support = [{i} for i in range(size)]
    boolean = np.eye(size, dtype=bool)
    receptive, jump = 1, 1
    trace = []
    for name, module in model.end_points.items():
        if type(module).__name__ == 'InceptionModule':
            # Union across output channels: 1x1 branches are contained in the
            # 3x3 branches; each following dense conv sees every input channel.
            branches = ((module.b0,), (module.b1a, module.b1b),
                        (module.b2a, module.b2b), (module.b3a, module.b3b))
        else:
            branches = ((module,),)
        outputs, bool_outputs, r_outputs, j_outputs = [], [], [], []
        for branch in branches:
            current, matrix, r, j = support, boolean, receptive, jump
            for op in branch:
                if type(op).__name__ == 'Unit3D':
                    k, stride = op._kernel_shape[axis], op._stride[axis]
                elif type(op).__name__ == 'MaxPool3dSamePadding':
                    k, stride = op.kernel_size[axis], op.stride[axis]
                else:
                    raise TypeError(type(op))
                current = set_step(current, k, stride)
                matrix = boolean_step(matrix, k, stride)
                r += (k-1)*j
                j *= stride
            outputs.append(current)
            bool_outputs.append(matrix)
            r_outputs.append(r)
            j_outputs.append(j)
        assert len(set(j_outputs)) == 1
        support = [set().union(*(x[i] for x in outputs)) for i in range(len(outputs[0]))]
        boolean = np.logical_or.reduce(bool_outputs)
        assert all(set(np.flatnonzero(boolean[i])) == item for i, item in enumerate(support))
        receptive, jump = max(r_outputs), j_outputs[0]
        trace.append({'endpoint': name, 'output_size': len(support), 'maximum_nominal_rf': receptive,
                      'jump': jump, 'min_support_size': min(map(len, support)),
                      'max_support_size': max(map(len, support))})
    n = len(support)
    bins = [list(range(math.floor(i*n/pooled_size), math.ceil((i+1)*n/pooled_size)))
            for i in range(pooled_size)]
    basis = F.adaptive_avg_pool1d(torch.eye(n, dtype=torch.float64)[None], pooled_size)[0].T
    assert all(torch.nonzero(basis[i] > 0).flatten().tolist() == b for i, b in enumerate(bins))
    pooled_support = [set().union(*(support[p] for p in indexes)) for indexes in bins]
    return {'input_size': size, 'trace': trace, 'adaptive_bins': bins,
            'pooled_support': [sorted(s) for s in pooled_support],
            'pooled_support_sizes': list(map(len, pooled_support)),
            'all_pooled_bins_have_full_axis_structural_support': all(len(s) == size for s in pooled_support),
            'set_boolean_methods_exact': True, 'adaptive_torch_basis_exact': True}


@torch.inference_mode()
def permutation_certificate():
    torch.manual_seed(20260915)
    raw = torch.randn(3, 32, 1024, dtype=torch.float64)
    v = torch.randn(3, 6, 512, dtype=torch.float64)
    t = torch.randn(4, 7, 512, dtype=torch.float64)
    vm = torch.zeros(3, 6, dtype=torch.long)
    tm = torch.ones(4, 7, dtype=torch.long)
    vm[0, -1] = 1
    tm[1, -2:] = 0
    grid = torch.arange(32).reshape(8, 4)
    permutations = {'within_window_cells': grid[:, [2, 0, 3, 1]].flatten(),
                    'window_order': grid[[3, 1, 6, 0, 7, 4, 2, 5]].flatten(),
                    'arbitrary_flat': torch.randperm(32)}
    result = {}
    for regime in ('R2', 'R3'):
        model = RawReadout(regime).double().eval()
        if regime == 'R2':
            model.scale.copy_(torch.tensor([1., -.7], dtype=torch.float64))
        else:
            model.pair[-1].weight.normal_(std=.2)
            model.pair[-1].bias.normal_(std=.2)
        original = model(raw, v, t, vm, tm)
        entry = {'permutations': {}, 'synthetic_nonzero_output_max': max(float(x.abs().max()) for x in original)}
        assert entry['synthetic_nonzero_output_max'] > 1e-6
        for label, permutation in permutations.items():
            transformed = model(raw[:, permutation], v, t, vm, tm)
            errors = [float((a-b).abs().max()) for a, b in zip(original, transformed)]
            assert max(errors) <= 1e-10
            entry['permutations'][label] = {'channel_max_abs_errors': errors, 'tolerance': 1e-10}
        replaced = raw.clone()
        replaced[:, 0] = torch.randn_like(replaced[:, 0])
        changed = model(replaced, v, t, vm, tm)
        differences = [float((a-b).abs().max()) for a, b in zip(original, changed)]
        assert min(differences) > 1e-12
        entry['content_replacement_channel_max_changes'] = differences
        result[regime] = entry
    return result


def main():
    path = OUT/'AS-C44-RAW-RELATION_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    started = time.time()
    implementation = ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py'
    sources = [implementation, ROOT/'methods/information_probe/raw_readout.py',
               ROOT/'methods/information_probe/scoring.py', ROOT/'methods/information_probe/cache_raw.py',
               ROOT/'shared/slr_common/features/i3d.py']
    report = {'status': 'running', 'pid': os.getpid(), 'experiment_id': 'AS-C44-RAW-RELATION',
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C44_protocol.md'),
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
              'scope': 'architecture plus synthetic CPU inputs; no dataset/feature/checkpoint load',
              'method_go': False, 'test_loaded': False, 'training_updates': 0}
    dump(path, report)
    try:
        spec = importlib.util.spec_from_file_location('as_c44_i3d', implementation)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.InceptionI3d(final_endpoint='Mixed_5c')
        recipe = ExtractionRecipe()
        report['axes'] = {label: axis_certificate(model, axis, size, pool)
                          for label, axis, size, pool in [('time', 0, recipe.clip_frames, 1),
                                                         ('height', 1, recipe.crop_size, 2),
                                                         ('width', 2, recipe.crop_size, 2)]}
        report['readout_invariance'] = permutation_certificate()
        report.update(status='completed', wall_seconds=time.time()-started,
                      limitations=['structural possible dependence, not active effective gradients or anatomical labels',
                                   'post-encoder token permutations, not real raw-video linguistic contrasts',
                                   'FP64 synthetic validation, not historical retrieval replay',
                                   'invariance applies to raw branch; I3D can encode local spatiotemporal relations inside each vector'])
        dump(path, report)
        print(json.dumps({**{k: v for k, v in report.items() if k != 'axes'},
                          'axis_summary': {k: {'last': v['trace'][-1], 'bins': v['adaptive_bins'],
                                               'support_sizes': v['pooled_support_sizes']}
                                           for k, v in report['axes'].items()}}, indent=2))
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, report)
        raise


if __name__ == '__main__':
    main()
