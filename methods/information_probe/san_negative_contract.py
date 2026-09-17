"""Synthetic SAN source-contract fixtures; no upstream imports or data access."""
import ast
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


def extract(path, owner, names, scope):
    tree = ast.parse(path.read_bytes())
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner]
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    ds = root / 'third_party/SAN/datasets.py'
    util = root / 'third_party/SAN/utils.py'
    scope = {'random': random, 'torch': torch, 'F': F, 'nn': nn}
    extract(ds, 'S2T_Dataset', ['generate_hard_negatives', 'collate_fn'], scope)
    generate = scope['generate_hard_negatives']
    extract(util, 'CrossEn', ['forward'], scope)
    loss = lambda scores: scope['forward'](None, scores, is_hard=True)
    random.seed(42)
    records = []

    repeated = generate(None, ['alpha'], {'alpha': ['beta']}, 5)
    assert repeated == [['beta'] * 5]
    records.append({'fixture': 'single_substitute', 'outputs': repeated})
    self_only = generate(None, ['alpha'], {'alpha': ['alpha']}, 5)
    assert self_only == [['alpha'] * 5]
    records.append({'fixture': 'adversarial_self_table', 'outputs': self_only})
    fallback = generate(None, ['alpha', 'beta'], {}, 5)
    assert fallback == [['beta'] * 5, ['alpha'] * 5]
    records.append({'fixture': 'other_caption_fallback', 'outputs': fallback})
    try:
        generate(None, ['alpha', 'alpha'], {}, 5)
    except IndexError:
        records.append({'fixture': 'empty_fallback_pool', 'exception': 'IndexError'})
    else:
        raise AssertionError('Expected empty-pool failure')

    # Intentionally mocked EDA/tokenizer. No inference about real EDA frequency.
    obj = SimpleNamespace(
        phase='train', args=SimpleNamespace(num_hard=5), hard_negative_table={},
        emb=SimpleNamespace(random_swap=lambda s: ' '.join(reversed(s.split()))),
        tokenizer=lambda strings, **kwargs: strings,
        generate_hard_negatives=lambda *a, **kw: generate(None, *a, **kw))
    batch = [(str(i), torch.ones(1, 1), torch.ones(1, dtype=torch.int), s)
             for i, s in enumerate(['alpha beta', 'beta alpha'])]
    with patch.object(random, 'random', return_value=1.):
        _, strings = scope['collate_fn'](obj, batch)
    assert strings[2:7] == [strings[0]] * 5
    assert strings[7:12] == [strings[1]] * 5
    records.append({'fixture': 'mocked_swap_collision', 'strings': strings,
                    'real_eda_executed': False, 'real_tokenizer_executed': False})

    # Rank-major gather and example-major flatten match the source layout.
    tags = torch.tensor([100 * r + 10 * b + k
                         for r in range(2) for b in range(2) for k in range(5)])
    score_tags = tags[None, :, None, None].expand(4, -1, 1, 1)
    grouped = score_tags.reshape(4, 4, 5, 1, 1)
    selected = grouped[torch.arange(4), torch.arange(4), :, 0, 0]
    assert torch.equal(selected, tags.reshape(4, 5))
    records.append({'fixture': 'ownership_indexing', 'selected_tags': selected.tolist(),
                    'actual_distributed_execution': False})

    shared = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    tied_loss = loss(shared.expand(1, 6))
    gradient, = torch.autograd.grad(tied_loss, shared)
    assert abs(float(tied_loss.detach()) - math.log(6)) < 1e-12
    assert abs(float(gradient)) < 1e-12
    records.append({'fixture': 'tied_shared_scores', 'loss': float(tied_loss.detach()),
                    'shared_gradient': float(gradient)})
    values = []
    for count in (1, 5):
        measured = float(loss(torch.tensor([[2.] + [1.] * count], dtype=torch.float64)))
        expected = math.log1p(count * math.exp(-1))
        assert abs(measured - expected) < 1e-12
        values.append({'multiplicity': count, 'loss': measured, 'formula': expected})
    assert values[1]['loss'] > values[0]['loss']
    records.append({'fixture': 'multiplicity', 'values': values})

    files = [ds, util, root / 'third_party/SAN/models.py', Path(__file__),
             root / 'docs/proposal7/evidence/autonomous_search/SAN_negative_contract_protocol.md']
    print(json.dumps({'status': 'completed', 'seed': 42, 'torch_version': torch.__version__,
                      'scope': 'eight synthetic source-contract fixtures only',
                      'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files},
                      'data_or_checkpoint_access': False, 'optimizer_updates': 0,
                      'method_go': False, 'records': records}, indent=2))


if __name__ == '__main__':
    main()
