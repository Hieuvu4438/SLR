import numpy as np

from .encoder_grid_probe import comparison, grid_aggregates


def test_grid_marginals_and_identity():
    grid = [[np.full((2, 2), i*10+j, np.float32) for j in range(3)] for i in range(3)]
    a = grid_aggregates(grid)
    assert np.array_equal(a['matched3'], a['mismatched6'])
    assert np.array_equal(a['matched3'], a['all9'])
    assert a['fixed_video_1337_mean_text3'][0, 0] == 11
    assert a['fixed_text_42_mean_video3'][0, 0] == 10
    grid[0][0][:] += 9
    b = grid_aggregates(grid)
    assert b['matched3'][0, 0]-a['matched3'][0, 0] == 3
    assert b['mismatched6'][0, 0] == a['mismatched6'][0, 0]
    assert b['all9'][0, 0]-a['all9'][0, 0] == 1


def test_comparison_scope_is_explicit():
    s = np.eye(3, dtype=np.float32)
    x = comparison(s, s, ['a', 'a', 'b'], 'offdiag minus diagonal')
    assert x['mean_pp'] == x['lower'] == x['upper'] == 0
    assert x['scope'].startswith('offdiag minus diagonal;')
