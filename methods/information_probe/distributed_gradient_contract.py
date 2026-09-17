"""Pinned CiCo AST, emulated gather transport, exact backward/gradient audit."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'third_party/SLRT/CiCo/CLCL/modules'
HASHES = {
    'until_module.py': '9fad15316a5fdc3133ffa2bd1e61b0bc94a9b4b5699af75662ccd1c1fd7538db',
    'modeling.py': 'aeaf646b720f4cf051950dcbac68347196bed4b1c856d9aa833b063ed09e9e7f',
}


def extract():
    nodes = []
    for filename, expected in HASHES.items():
        raw = (SOURCE / filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        tree = ast.parse(raw)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name in ('AllGather', 'CrossEn'):
                    nodes.append(node)
                if node.name == 'CLIP4Clip':
                    nodes.extend(n for n in node.body if isinstance(n, ast.FunctionDef)
                                 and n.name == 'flip_similarity_softmax')
    scope = {'torch': torch, 'nn': nn, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned_cico_ast', 'exec'), scope)
    return scope


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    scope = extract()
    x, y, ya = [torch.randn(8, n, 4, dtype=torch.float64) for n in (2, 3, 3)]
    initial = [torch.randn(4, 5, dtype=torch.float64) for _ in range(2)]
    initial.append(torch.tensor(.7, dtype=torch.float64))

    def replica(world, rank, mode):
        params = [p.clone().requires_grad_() for p in initial]
        wv, wt, scale = params
        features = [x @ wv, y @ wt, ya @ wt]
        gathered = []
        for full in features:
            chunks = [c.detach() for c in full.chunk(world)]

            def transport(output, local):
                for out, chunk in zip(output, chunks):
                    out.copy_(chunk)

            local = full.chunk(world)[rank]
            with patch.object(torch.distributed, 'all_gather', transport):
                out = scope['AllGather'].apply(local, SimpleNamespace(world_size=world, rank=rank))
            if mode == 'feature_scaled':
                out.register_hook(lambda g: g * world)
            gathered.append(out)
        visual, text, augmented = gathered
        obj = SimpleNamespace(training=False, clip=SimpleNamespace(logit_scale=scale))
        a, b = scope['flip_similarity_softmax'](
            obj, text, visual, torch.ones(8, 3), torch.zeros(8, 2),
            sequence_hidden_aug=augmented, text_mask_aug=torch.ones(8, 3))
        ce = scope['CrossEn']()
        loss = (ce(a) + ce(a.T) + ce(b) + ce(b.T)) / 4
        if mode == 'loss_scaled':
            loss = loss * world
        grads = torch.autograd.grad(loss, params)
        return float(loss.detach()), grads

    reference_loss, reference = replica(1, 0, 'source')
    assert all(g.norm() > 1e-8 for g in reference)
    records = []
    for world in (1, 2, 4):
        for mode in ('source', 'feature_scaled', 'loss_scaled'):
            replicas = [replica(world, rank, mode) for rank in range(world)]
            average = [torch.stack([r[1][j] for r in replicas]).mean(0) for j in range(3)]
            factors = ([1/world, 1/world, 1] if mode == 'source' else
                       [1, 1, world] if mode == 'loss_scaled' else [1, 1, 1])
            errors = [float((g - f*r).abs().max()) for g, f, r in zip(average, factors, reference)]
            expected_loss = reference_loss * (world if mode == 'loss_scaled' else 1)
            loss_error = max(abs(r[0] - expected_loss) for r in replicas)
            assert max(errors + [loss_error]) < 1e-10
            records.append({'world': world, 'mode': mode, 'expected_factors': factors,
                            'observed_norm_ratios': [float(g.norm()/r.norm()) for g, r in zip(average, reference)],
                            'max_coordinate_errors': errors, 'loss_error': loss_error})
    return {'status': 'completed', 'scope': 'synthetic source-backward fixture; transport and DDP average emulated',
            'actual_ddp_execution': False, 'benchmark_or_checkpoint_access': False,
            'optimizer_updates': 0, 'method_go': False, 'torch': torch.__version__,
            'source_sha256': HASHES, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reference_loss': reference_loss, 'records': records}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
