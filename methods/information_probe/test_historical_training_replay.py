import ast
import numpy as np
import torch
from .historical_training_replay import archived, check_sources, definitions, exact_compare, historical_function


def test_recursive_exact_comparison():
    x = {'a': torch.tensor([1., 2.]), 'b': [np.array([3, 4]), (1, 'ok')]}
    assert exact_compare(x, {'a': x['a'].clone(), 'b': [np.array([3, 4]), (1, 'ok')]}) == []
    assert exact_compare(x, {**x, 'a': torch.tensor([1., 2.0001])})[0]['path'] == 'root/a'
    assert exact_compare([1], (1,))
    assert exact_compare({'a': 1}, {'a': 1, 'b': 2})[0]['kind'] == 'missing_key'


def test_archived_function_and_active_dependencies():
    checks = check_sources()
    assert len(checks) == 13
    source = archived('elsc/train.py')
    node = definitions(source)['train']
    ns = {'dict': dict, 'Any': object, 'Path': object, 'torch': torch}
    fn = historical_function(ns)
    assert fn.__name__ == 'train'
    assert fn.__code__.co_filename.startswith('git:39449e18')
    assert not any(isinstance(n, ast.Constant) and n.value == 'initialization_epoch_minus_one_and_all_post_epoch_checkpoints'
                   for n in ast.walk(node))
