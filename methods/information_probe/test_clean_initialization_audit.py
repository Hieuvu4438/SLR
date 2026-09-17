from methods.information_probe.clean_initialization_audit import source_fold, component_sizes


def test_components_connect_source_and_exact_text_transitively():
    assert component_sizes(['a', 'a', 'b', 'c'], [(1,), (2,), (2,), (3,)]) == [3, 1]
    assert component_sizes(['a', 'b', 'c'], [(1,), (2,), (3,)]) == [1, 1, 1]


def test_source_fold_deterministic_and_bounded():
    assert source_fold('source-one') == source_fold('source-one')
    assert all(0 <= source_fold(str(i)) < 5 for i in range(100))
