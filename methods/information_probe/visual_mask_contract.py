"""Source-only visual padding/key-mask information-flow certificate."""
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def run():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    source = ROOT / 'third_party/SLRT/CiCo/CLCL/modules/module_clip.py'
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    assert source_hash == '8fae51a298872c9d9e07e898a7f09a117cdfbd21a0fc51c2f3a608af2ee6d2d5'
    names = {'LayerNorm', 'QuickGELU', 'VResidualAttentionBlock', 'VTransformer', 'FeatureTransformer'}
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = {'torch': torch, 'nn': nn, 'OrderedDict': OrderedDict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    encoder = scope['FeatureTransformer'](224, 1, 8, 2, 2, 6, 4).eval()
    mask = torch.tensor([[1, 0, 0, 1, 1]])
    base = torch.randn(1, 1024, 4, 1)
    base[:, :, 2:] = 0
    pad_changed = base.clone()
    pad_changed[:, :, 2:] = 3 * torch.randn_like(pad_changed[:, :, 2:])
    valid_changed = base.clone()
    valid_changed[:, :, :2] += torch.randn_like(valid_changed[:, :, :2])

    def encode(x):
        return encoder.ln_post(encoder(x, mask)) @ encoder.proj

    with torch.no_grad():
        baseline = encode(base)
        other_pad = encode(pad_changed)
        other_valid = encode(valid_changed)
    valid = mask[0] == 0
    masked = ~valid
    valid_error = float((baseline[:, valid] - other_pad[:, valid]).abs().max())
    masked_change = float((baseline[:, masked] - other_valid[:, masked]).abs().max())
    x = base.clone().requires_grad_()
    output = encode(x)
    valid_probe = torch.randn_like(output[:, valid])
    masked_probe = torch.randn_like(output[:, masked])
    g_valid = torch.autograd.grad((output[:, valid] * valid_probe).sum(), x, retain_graph=True)[0]
    g_masked = torch.autograd.grad((output[:, masked] * masked_probe).sum(), x)[0]
    pad_gradient = float(g_valid[:, :, 2:].abs().max())
    readout_gradient = float(g_masked[:, :, :2].norm())
    assert valid_error <= 1e-6 and pad_gradient <= 1e-9
    assert masked_change > 1e-6 and readout_gradient > 1e-6
    return {'status': 'completed', 'source_sha256': source_hash,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'torch': torch.__version__, 'dtype': 'float32', 'seed': 42,
            'valid_output_change_when_pad_inputs_change_max': valid_error,
            'valid_output_probe_pad_input_gradient_max': pad_gradient,
            'masked_output_change_when_valid_inputs_change_max': masked_change,
            'masked_output_probe_valid_input_gradient_norm': readout_gradient,
            'scope': 'random two-layer source encoder, fixed sum/2D shape; not trained CiCo',
            'benchmark_or_checkpoint_access': False, 'method_go': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
