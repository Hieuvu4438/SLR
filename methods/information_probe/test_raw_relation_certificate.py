import unittest
import numpy as np

from .raw_relation_certificate import boolean_step, local_windows, set_step


class RawRelationTests(unittest.TestCase):
    def test_same_padding_odd_even(self):
        self.assertEqual(local_windows(5, 3, 2), [[0, 1], [1, 2, 3], [3, 4]])
        self.assertEqual(local_windows(6, 3, 2), [[0, 1, 2], [2, 3, 4], [4, 5]])

    def test_independent_support_methods(self):
        for n in (2, 3, 7, 16):
            support, boolean = [{i} for i in range(n)], np.eye(n, dtype=bool)
            for kernel, stride in [(3, 1), (2, 2), (3, 1)]:
                support = set_step(support, kernel, stride)
                boolean = boolean_step(boolean, kernel, stride)
                self.assertEqual(support, [set(np.flatnonzero(row)) for row in boolean])


if __name__ == '__main__':
    unittest.main()
