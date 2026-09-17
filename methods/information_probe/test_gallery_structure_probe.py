import unittest
import numpy as np

from .gallery_structure_probe import filter_queries, neighbors, relabel


class GalleryStructureTests(unittest.TestCase):
    def test_graph_self_exclusion_and_stable_ties(self):
        graph = neighbors(np.ones((4, 3)), k=2)
        np.testing.assert_array_equal(graph, [[1, 2], [0, 2], [0, 1], [0, 1]])

    def test_query_independence_identity_and_constant(self):
        graph = np.array([[1, 2], [0, 2], [0, 1]])
        s = np.array([[1, 4, 9], [2, 5, 8]], dtype=np.float32)
        np.testing.assert_array_equal(filter_queries(s, graph, 0), s)
        for beta in (.5, -.5):
            together = filter_queries(s, graph, beta)
            alone = np.concatenate([filter_queries(x[None], graph, beta) for x in s])
            np.testing.assert_array_equal(together, alone)
            np.testing.assert_array_equal(filter_queries(np.ones_like(s), graph, beta), np.ones_like(s))
            expected = s + beta * (np.array([[6.5, 5, 2.5], [6.5, 5, 3.5]]) - s)
            np.testing.assert_array_equal(together, expected)

    def test_relabel_conjugates_graph(self):
        graph = np.array([[1, 2], [2, 3], [3, 0], [0, 1]])
        p = np.array([3, 0, 2, 1])
        s = np.array([[1, 3, 5, 8]], dtype=np.float32)
        np.testing.assert_array_equal(filter_queries(s[:, p], relabel(graph, p), .5),
                                      filter_queries(s, graph, .5)[:, p])


if __name__ == '__main__':
    unittest.main()
