import numpy as np

from methods.information_probe.lexical_contrast_audit import words, groups_with_order_contrasts, margins_from_scores


def test_word_multiset_keeps_repetitions_and_excludes_identical_orders():
    seq = [words(x) for x in ("North, then south!", "south then north", "north then south", "north north south", "south north north", "north south")]
    assert groups_with_order_contrasts(seq) == [[0, 1, 2], [3, 4]]
    assert words("Don't lose Süd-West.") == ("don't", 'lose', 'süd', 'west')


def test_four_margins_orientation_and_numerical_band():
    s = np.array([[4., 2.], [3., 5.]])
    x = margins_from_scores(s, [(0, 1)])[0]
    assert x['directional_margins'] == {'V2T_a': 2., 'V2T_b': 2., 'T2V_a': 1., 'T2V_b': 3.}
    assert x['all_four_strict']
    assert not margins_from_scores(np.ones((2, 2)), [(0, 1)])[0]['all_four_strict']
