from research.slret_dataset_first.tools.analyze_structure import (
    lexical_units,
    ngrams,
    normalize_text,
)


def test_normalize_text_is_nfkc_casefolded_and_space_stable():
    assert normalize_text("  Straße\tＡ  ") == "strasse a"


def test_lexical_units_supports_chinese_and_latin():
    assert lexical_units("你们好！") == ("你", "们", "好")
    assert lexical_units("It's partly-cloudy.") == ("it's", "partly-cloudy")


def test_ngrams_are_complete_sets():
    tokens = ("a", "b", "c")
    assert ngrams(tokens, 1) == {("a",), ("b",), ("c",)}
    assert ngrams(tokens, 2) == {("a", "b"), ("b", "c")}
    assert ngrams(tokens, 4) == set()
