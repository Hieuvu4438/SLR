import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from seds_pair_audit import same_tree


class StateComparisonTests(unittest.TestCase):
    def test_nested_rng(self):
        a = {'python': (3, (1, 2, 3), None), 'numpy': ('MT19937', np.array([2, 3])),
             'torch': torch.tensor([4, 5], dtype=torch.uint8), 'cuda': [torch.tensor([6])]}
        b = {'python': (3, (1, 2, 3), None), 'numpy': ('MT19937', np.array([2, 3])),
             'torch': torch.tensor([4, 5], dtype=torch.uint8), 'cuda': [torch.tensor([6])]}
        self.assertTrue(same_tree(a, b))
        b['cuda'][0][0] = 7
        self.assertFalse(same_tree(a, b))

    def test_type_and_dtype_mismatches(self):
        self.assertFalse(same_tree([1], (1,)))
        self.assertFalse(same_tree(torch.tensor([1]), torch.tensor([1.])))
        self.assertFalse(same_tree(np.array([1], dtype=np.float32), np.array([1], dtype=np.float64)))

    def test_key_and_length_mismatches(self):
        self.assertFalse(same_tree({'a': 1}, {'b': 1}))
        self.assertFalse(same_tree([1], [1, 2]))
        self.assertFalse(same_tree({'a': 1}, {'a': 1, 'b': 2}))


if __name__ == '__main__':
    unittest.main()
