from __future__ import annotations

from elsc.data.word_offsets import eligible_word_units, encode_with_offsets, replace_canonical_span


def test_ids_match_upstream_with_unicode_html_and_punctuation(cico_tokenizer, cico_cleaners):
    basic_clean, whitespace_clean = cico_cleaners
    encoded = encode_with_offsets(
        "  Grüße &amp; schönes Wetter, heute!  ",
        cico_tokenizer,
        32,
        basic_clean=basic_clean,
        whitespace_clean=whitespace_clean,
    )
    assert encoded.canonical == "grüße & schönes wetter, heute!"
    assert encoded.token_ids[0] == cico_tokenizer.encoder["<|startoftext|>"]
    assert encoded.token_ids[sum(encoded.input_mask) - 1] == cico_tokenizer.encoder["<|endoftext|>"]
    assert all(word.fully_retained for word in encoded.words)


def test_repeated_word_is_not_auxiliary_eligible(cico_tokenizer, cico_cleaners):
    basic_clean, whitespace_clean = cico_cleaners
    encoded = encode_with_offsets(
        "rain, rain and sun",
        cico_tokenizer,
        16,
        basic_clean=basic_clean,
        whitespace_clean=whitespace_clean,
    )
    eligible = eligible_word_units(encoded, stopwords={"and"})
    assert [word.surface for word in eligible] == ["sun"]


def test_partial_bpe_subsampling_excludes_word_but_keeps_caption(cico_tokenizer, cico_cleaners):
    basic_clean, whitespace_clean = cico_cleaners
    text = "extraordinarily meteorological precipitation continues tomorrow morning"
    encoded = encode_with_offsets(
        text, cico_tokenizer, 6, basic_clean=basic_clean, whitespace_clean=whitespace_clean
    )
    assert encoded.was_subsampled
    assert len(encoded.token_ids) == 6
    assert any(not word.fully_retained for word in encoded.words)
    assert all(word.fully_retained for word in eligible_word_units(encoded))


def test_replacement_changes_only_the_selected_occurrence(cico_tokenizer, cico_cleaners):
    basic_clean, whitespace_clean = cico_cleaners
    encoded = encode_with_offsets(
        "rain then rain",
        cico_tokenizer,
        16,
        basic_clean=basic_clean,
        whitespace_clean=whitespace_clean,
    )
    first = encoded.words[0]
    replaced = replace_canonical_span(encoded.canonical, first.canonical_char_span, "snow")
    assert replaced == "snow then rain"
