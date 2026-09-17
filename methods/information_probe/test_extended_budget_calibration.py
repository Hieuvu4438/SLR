import ast
import copy

from .extended_budget_calibration import transformed_tree
from .freeze_training_comparison import SOURCE


def test_registered_changes_are_reversible_and_preserve_metrics():
    source = SOURCE.read_text()
    tree, changes = transformed_tree(source)
    restored = copy.deepcopy(tree)
    old = {(c['line'], c['column']): c['old'] for c in changes}
    for node in ast.walk(restored):
        if isinstance(node, ast.Constant) and (node.lineno, node.col_offset) in old:
            node.value = old[node.lineno, node.col_offset]
    assert ast.dump(restored) == ast.dump(ast.parse(source))
    for name in ('compact_metrics', 'adequacy', 'evaluate'):
        before = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(before) == ast.dump(after)


def test_schedule_is_proportionally_stretched():
    tree, _ = transformed_tree(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lr_factor')
    import math
    scope = {'math': math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'schedule', 'exec'), scope)
    lr = scope['lr_factor']
    assert lr(0) == 0 and lr(880) == 1 and lr(8800) == 0
    for a, b in ((2200, 250), (4400, 500), (8800, 1000)):
        assert abs(lr(a)-lr(b, total=1000, warmup=100)) < 1e-15
    assert lr(100) != lr(100, total=1000, warmup=100)


def test_absolute_adequacy_cannot_be_replaced_by_relative_gain():
    from .validate_extended_budget import independent_gate
    initial = {'held': {'mean_R1': 0}}
    final = {'fit': {d: {'R1': 90} for d in ('T2V', 'V2T')},
             'held': {d: {'R1': 49, 'strict_different_text_error_n': 100} for d in ('T2V', 'V2T')}}
    final['held']['mean_R1'] = 49
    assert not independent_gate(initial, final)['diagnostic_adequacy']
    for d in ('T2V', 'V2T'):
        final['held'][d]['R1'] = 50
    final['held']['mean_R1'] = 50
    assert independent_gate(initial, final)['diagnostic_adequacy']
    final['held']['V2T']['strict_different_text_error_n'] = 49
    assert not independent_gate(initial, final)['diagnostic_adequacy']


def test_recovery_changes_only_six_output_literal_occurrences():
    from .recover_extended_budget import HARNESS, transform
    tree, changes = transform(HARNESS.read_text())
    assert len(changes) == 6
    assert len({x['old'] for x in changes}) == 3
    assert all(x['new'] == x['old'].replace('AS-C42-BUDGET', 'AS-C42-BUDGET-attempt2') for x in changes)
    before = ast.parse(HARNESS.read_text())
    for name in ('transformed_tree', 'main'):
        old = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        old_calls = [ast.dump(n.func) for n in ast.walk(old) if isinstance(n, ast.Call)]
        new_calls = [ast.dump(n.func) for n in ast.walk(new) if isinstance(n, ast.Call)]
        assert old_calls == new_calls
