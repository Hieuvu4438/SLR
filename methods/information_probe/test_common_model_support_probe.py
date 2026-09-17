from .common_model_support_probe import subset_indexes


def test_subset_reproducible_and_original_order():
    ids = [f'row{i}' for i in range(30)]
    a = subset_indexes(ids, 0, 10)
    assert a == subset_indexes(ids, 0, 10) and a == sorted(set(a)) and len(a) == 10
    assert a != subset_indexes(ids, 1, 10)
