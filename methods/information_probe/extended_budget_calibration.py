"""One preregistered AS-C20 budget extension; no recipe or method novelty."""
import ast
from collections import Counter
import contextlib
import hashlib
import json

import numpy as np
import torch

from .common import ART, ROOT, dump, sha
from .clean_initialization_audit import OUT
from .freeze_training_comparison import SOURCE, SOURCE_SHA


def transformed_tree(source):
    tree = ast.parse(source)
    changes = []
    strings = {
        'AS-C20-TRAIN_run.json': 'AS-C42-BUDGET_run.json',
        'AS-C20': 'AS-C42-BUDGET',
        'AS-C20-TRAIN': 'AS-C42-BUDGET',
        'AS-C20_protocol.md': 'AS-C42_protocol.md',
        'final_step1000.pt': 'final_step8800.pt',
        'preregistered 3600-second hard timeout': 'preregistered 7200-second hard timeout',
        'AS-C20 outputs already exist; never overwrite a prior attempt':
            'AS-C42 outputs already exist; never overwrite a prior attempt',
    }
    # The other literal100 is the percentage multiplier: never replace it.
    lr = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lr_factor')
    warmup = lr.args.defaults[-1]
    assert warmup.value == 100
    replacements = {1000: 8800, 250: 2200, 500: 4400, 3600: 7200}
    counts = Counter(n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and type(n.value) is int and n.value in replacements)
    assert counts == Counter({1000: 5, 250: 1, 500: 1, 3600: 1}), counts
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        old = node.value
        if node is warmup:
            new = 880
        elif type(old) is int and old in replacements:
            new = replacements[old]
        elif isinstance(old, str) and old in strings:
            new = strings[old]
        else:
            continue
        node.value = new
        changes.append({'line': node.lineno, 'column': node.col_offset, 'old': old, 'new': new})
    assert len(changes) == 16
    return tree, changes


def main():
    assert sha(SOURCE) == SOURCE_SHA
    dest = ART/'AS-C42-BUDGET'
    target = OUT/'AS-C42-BUDGET_run.json'
    console = ART/'AS-C42-BUDGET-console.log'
    if dest.exists() or target.exists() or console.exists():
        raise FileExistsError('AS-C42 outputs exist; inspect state instead of restarting')
    original_path = OUT/'AS-C20-TRAIN_run.json'
    original = json.loads(original_path.read_text())
    control = json.loads((OUT/'AS-C30-CONTROL_run.json').read_text())
    assert original['status'] == control['status'] == 'completed'
    assert control['comparison']['exact_replay']
    tree, changes = transformed_tree(SOURCE.read_text())
    namespace = {'__name__': 'methods.information_probe._asc42_budget',
                 '__package__': 'methods.information_probe', '__file__': str(SOURCE)}
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    provenance = {
        'harness_sha256': sha(__file__), 'source_sha256': SOURCE_SHA,
        'parent_run_sha256': sha(original_path),
        'reproduced_control_sha256': sha(OUT/'AS-C30-CONTROL_run.json'),
        'transformed_ast_sha256': hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        'literal_changes': changes, 'total_updates': 8800, 'warmup_updates': 880,
        'full_training_passes': 200, 'upstream_reproduction': False,
        'comparison_scope': 'joint budget and proportionally stretched schedule; not identical-prefix continuation',
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0)},
        'helper_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT/'methods/information_probe/clean_initialization_audit.py',
            ROOT/'methods/information_probe/scoring.py',
            ROOT/'shared/slr_common/data/cico_dataset.py',
            ROOT/'shared/slr_common/data/tokenize.py',
            ROOT/'shared/slr_common/upstream/cico_bridge.py')},
    }
    initialization_check = None

    def record(path, result):
        nonlocal initialization_check
        result.update(provenance)
        if result['evaluations'] and initialization_check is None:
            entry = result['evaluations'][0]
            assert entry['step'] == 0 and result['updates'] == 0
            checks = {}
            for split in ('fit', 'held'):
                a, b = ART/'AS-C20'/f'{split}_step0.npy', dest/f'{split}_step0.npy'
                checks[split] = {'array_equal': bool(np.array_equal(np.load(a), np.load(b))),
                                 'sha256_equal': sha(a) == sha(b)}
            initialization_check = {'score_checks': checks, 'evaluation_exact': entry == original['evaluations'][0]}
            if not initialization_check['evaluation_exact'] or not all(all(x.values()) for x in checks.values()):
                result['initialization_check'] = initialization_check
                dump(path, result)
                raise AssertionError('AS-C42 initialization mismatch; training forbidden')
        result['initialization_check'] = initialization_check
        if result['status'] == 'completed':
            assert result['updates'] == 8800
            assert [e['step'] for e in result['evaluations']] == [0, 2200, 4400, 8800]
            result['short_budget_comparison'] = {
                'short_held_mean_R1': original['evaluations'][-1]['held']['mean_R1'],
                'extended_held_mean_R1': result['evaluations'][-1]['held']['mean_R1'],
                'delta_mean_R1_pp': result['evaluations'][-1]['held']['mean_R1']-original['evaluations'][-1]['held']['mean_R1'],
                'short_adequacy': original['adequacy'],
            }
        dump(path, result)

    namespace['dump'] = record
    with console.open('x', buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        namespace['main']()


if __name__ == '__main__':
    main()
