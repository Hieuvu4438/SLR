from .single_word_contrast_audit import bilingual_pairs, one_substitution_pairs, summarize_margins


def test_exactly_one_substitution_with_repeats():
    seq = [tuple(x.split()) for x in ('a b c', 'a x c', 'a b c', 'a c b', 'a x z', 'a b', '')]
    assert one_substitution_pairs(seq) == {(0, 1): 1, (1, 2): 1, (1, 4): 2}


def test_bilingual_intersection_and_different_positions():
    en = [('a', 'b'), ('a', 'c'), ('a', 'd')]
    de = [('x', 'y'), ('z', 'y'), ('x', 'w', 'y')]
    a, b, joint = bilingual_pairs(en, de)
    assert len(a) == 3 and joint == [(0, 1)]
    assert a[0, 1] == 1 and b[0, 1] == 0
    empty = summarize_margins([])
    assert empty['minimum_margin'] is None and empty['directional_margin_n'] == 0
