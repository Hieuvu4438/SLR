import ast
import torch

from .freeze_training_comparison import SOURCE, SOURCE_SHA, FROZEN, freeze_tables, transformed_tree
from .common import sha


def test_transform_changes_only_four_literals():
    assert sha(SOURCE) == SOURCE_SHA
    original = ast.parse(SOURCE.read_text())
    for arm in ('control', 'freeze'):
        tree, mapping = transformed_tree(SOURCE.read_text(), arm)
        count = 0
        for node, old in zip(ast.walk(tree), ast.walk(original), strict=True):
            if isinstance(old, ast.Constant) and isinstance(old.value, str) and old.value in mapping:
                assert node.value == mapping[old.value]
                node.value = old.value
                count += 1
        assert count == 4
        assert ast.dump(tree) == ast.dump(original)


def test_freeze_only_two_keys():
    class Tiny:
        def __init__(self):
            self.params = {k: torch.nn.Parameter(torch.ones(2)) for k in (*FROZEN, 'other.weight')}
        def named_parameters(self):
            return self.params.items()
    control, freeze = Tiny(), Tiny()
    assert freeze_tables(control, 'control')['frozen_parameter_n'] == 0
    assert freeze_tables(freeze, 'freeze')['frozen_parameter_n'] == 4
    assert freeze.params['other.weight'].requires_grad
    optimizer = torch.optim.AdamW(freeze.params.values(), lr=.1, weight_decay=.1)
    freeze.params['other.weight'].sum().backward()
    optimizer.step()
    for k in FROZEN:
        assert torch.equal(freeze.params[k], torch.ones(2))
