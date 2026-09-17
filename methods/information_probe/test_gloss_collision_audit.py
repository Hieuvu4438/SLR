import pytest
from .gloss_collision_audit import exact_pairs


def test_exact_sequence_not_bag_or_near_match():
    pairs, groups = exact_pairs([('A', 'B'), ('B', 'A'), ('A', 'B'), ('A',), ('A', 'B')])
    assert pairs == [(0, 2), (0, 4), (2, 4)]
    assert len(groups) == 3


def test_empty_gloss_is_not_equivalence():
    with pytest.raises(ValueError, match='Empty gloss'):
        exact_pairs([(), ()])
