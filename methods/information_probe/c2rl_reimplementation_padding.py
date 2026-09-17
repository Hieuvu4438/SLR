"""Pinned third-party C2RL-derived kernel fixture, never original-code parity."""
import ast
import hashlib
import json
import math
import urllib.request

import torch

from .common import ROOT, dump, sha

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'
COMMIT = 'f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13'
SOURCE_SHA = '81edd2f15e09ac5d388303a142798ac425c04896edffc6ed036d03d1b79e54a2'
URL = f'https://raw.githubusercontent.com/ozgemercanoglu/sltbaselines/{COMMIT}/models/models.py'


class CPUOnly(ast.NodeTransformer):
    def __init__(self):
        self.replaced = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            assert not node.args and not node.keywords
            self.replaced += 1
            return node.func.value
        return node


def neutral_pool(c, valid_n, total_n, tau=.07):
    mass = valid_n * math.exp(c/tau)
    return c * mass / (mass + total_n-valid_n)


def main():
    torch.set_num_threads(1)
    output = OUT / 'C2RL-REIMPLEMENTATION-PADDING.json'
    if output.exists():
        raise FileExistsError(output)
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    tree = ast.parse(raw)
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CiCo']
    method, = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name == 'cross_lingual_similarity_v2']
    original_ast_sha = hashlib.sha256(ast.dump(method).encode()).hexdigest()
    adapter = CPUOnly()
    method = adapter.visit(method)
    assert adapter.replaced == 4
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 URL, 'exec'), scope)
    kernel = scope['cross_lingual_similarity_v2']
    visual = torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]]], dtype=torch.float64)
    vm = torch.ones(2, 1, dtype=torch.long)
    results, tensors = {}, {}
    for total in (2, 32):
        text = torch.tensor([0., 0., 1.], dtype=torch.float64).expand(2, total, 3).clone()
        mask = torch.zeros(2, total, dtype=torch.long)
        for row, (c, n) in enumerate(((.21, 1), (.19, 2))):
            text[row, :n] = torch.tensor([c, math.sqrt(1-c*c), 0.], dtype=torch.float64)
            mask[row, :n] = 1
        a, b, loss = kernel(None, visual, text, vm, mask, 1.)
        predicted = torch.tensor([neutral_pool(.21, 1, total), neutral_pool(.19, 2, total)],
                                 dtype=torch.float64)
        error = float((a[0]-predicted).abs().max())
        assert error < 1e-12
        sim = torch.einsum('bid,cjd->bcij', visual, text)
        masked = sim.masked_fill(~mask.bool()[None, :, None], -torch.inf)
        # Independent inner-exclusion reference; NOT a proposed repository fix.
        ref = (sim * (masked/.07).softmax(-1)).sum(-1).squeeze(-1)
        results[str(total)] = {
            'I2T': a.tolist(), 'T2I': b.tolist(), 'source_loss': float(loss),
            'query0_A_minus_B': float(a[0, 0]-a[0, 1]),
            'query0_prefers_A': bool(a[0, 0] > a[0, 1]),
            'closed_form_max_error': error, 'inner_exclusion_reference': ref.tolist(),
        }
        tensors[total] = (a, b, ref)
    t2i_delta = float((tensors[2][1]-tensors[32][1]).abs().max())
    reference_delta = float((tensors[2][2]-tensors[32][2]).abs().max())
    assert t2i_delta < 1e-12 and reference_delta < 1e-12
    report = {
        'status': 'completed', 'commit': COMMIT, 'source_url': URL,
        'source_sha256': SOURCE_SHA, 'source_method_ast_sha256': original_ast_sha,
        'cpu_adaptation': {'removed_zero_argument_cuda_calls': adapter.replaced},
        'code_sha256': sha(__file__),
        'protocol_sha256': sha(OUT / 'C2RL_reimplementation_padding_protocol.md'),
        'torch_version': torch.__version__, 'dtype': 'float64', 'fixtures': results,
        'T2I_padding_delta': t2i_delta, 'inner_exclusion_padding_delta': reference_delta,
        'I2T_query0_order_reverses': results['2']['query0_prefers_A'] != results['32']['query0_prefers_A'],
        'dataset_model_checkpoint_access': False, 'method_go': False,
        'limits': ['Independent SLT reimplementation, not original C2RL retrieval release.',
                   'Fixed synthetic valid vectors and orthogonal padding.',
                   'No semantic relevance or trained-model harm measured.',
                   'CPU AST device adaptation; no full runtime/GPU parity.'],
    }
    dump(output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
