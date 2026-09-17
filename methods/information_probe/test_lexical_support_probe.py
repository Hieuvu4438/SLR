from .lexical_support_probe import nearest_jaccard


def test_jaccard_and_source_exclusion():
    q, r = [{1, 2}, {3}], [{1, 2}, {2, 3}, {4}]
    out = nearest_jaccard(q, r, ['a', 'd'], ['a', 'b', 'c'])
    assert out[0]['best_jaccard'] == 1 and out[0]['reference_index'] == 0
    assert out[0]['cross_source_best_jaccard'] == 1/3 and out[0]['cross_source_reference_index'] == 1
    assert out[1]['best_jaccard'] == .5 and out[1]['reference_index'] == 1


def test_empty_sets_and_no_cross_source_reference():
    out = nearest_jaccard([set()], [set()], ['a'], ['a'])[0]
    assert out['best_jaccard'] == 1
    assert out['cross_source_best_jaccard'] is None
