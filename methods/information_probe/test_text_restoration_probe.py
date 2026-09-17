from .text_omission_inventory import selected_indexes
from .text_restoration_probe import repeat_retained


def test_matched_control_keeps_positions_and_excludes_omitted_identity():
    tokens = list(range(36))
    keep = selected_indexes(36)
    result = repeat_retained(tokens)
    assert len(result) == len(tokens)
    assert all(result[k] == tokens[k] for k in keep)
    assert set(result) == set(keep)
    assert repeat_retained([1, 2]) == [1, 2]
    assert repeat_retained([]) == []
