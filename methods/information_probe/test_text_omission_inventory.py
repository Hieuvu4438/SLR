from .text_omission_inventory import new_collisions, selected_indexes


def test_selection_preserves_boundary_and_capacity():
    assert selected_indexes(0) == []
    assert selected_indexes(30) == list(range(30))
    x = selected_indexes(31)
    assert len(x) == 30 and x[0] == 0 and x[-1] == 30
    assert len(set(x)) == 30
    assert selected_indexes(90)[-1] == 89


def test_collision_requires_distinct_full_sequences():
    assert new_collisions([[1, 2], [1, 2]], [[1], [1]]) == []
    assert new_collisions([[1, 2], [1, 3], [4]], [[1], [1], [4]]) == [[0, 1]]
