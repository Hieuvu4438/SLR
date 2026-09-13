from __future__ import annotations

import numpy as np

from method1.token_spans import (
    auxiliary_edit_eligible,
    canonicalize,
    replace_occurrence,
    tokenize_with_spans,
)


def test_canonicalization_and_repeated_occurrences(cico_tokenizer) -> None:
    encoded = tokenize_with_spans(
        "  Café&nbsp; rain RAIN!  ",
        text_uid="ph:train:x",
        tokenizer=cico_tokenizer,
    )
    assert encoded.canonical_text == "café rain rain!"
    rain = [span for span in encoded.lexical_spans if span.surface == "rain"]
    assert len(rain) == 2
    assert rain[0].occurrence_uid != rain[1].occurrence_uid


def test_contractions_and_hyphenated_units_are_excluded(cico_tokenizer) -> None:
    encoded = tokenize_with_spans(
        "can't state-of-the-art plain",
        text_uid="h2:train:x",
        tokenizer=cico_tokenizer,
    )
    assert [span.surface for span in encoded.lexical_spans] == ["plain"]


def test_changed_bpe_length_maps_only_edited_occurrence(cico_tokenizer) -> None:
    positive = tokenize_with_spans(
        "rain then rain",
        text_uid="ph:train:x",
        tokenizer=cico_tokenizer,
    )
    second = [span for span in positive.lexical_spans if span.surface == "rain"][1]
    negative, replacement = replace_occurrence(
        positive, second, "thunderstorm", tokenizer=cico_tokenizer
    )
    assert negative.canonical_text == "rain then thunderstorm"
    assert negative.canonical_text[replacement.char_span[0] : replacement.char_span[1]] == "thunderstorm"
    assert canonicalize("rain then rain")[: second.char_span[0]] == negative.canonical_text[: second.char_span[0]]
    assert auxiliary_edit_eligible(positive, second, negative, replacement) == (True, None)


def test_long_text_matches_uniform_bpe_and_disables_auxiliary(cico_tokenizer) -> None:
    text = " ".join(f"word{index}" for index in range(80))
    encoded = tokenize_with_spans(
        text,
        text_uid="h2:train:long",
        tokenizer=cico_tokenizer,
        max_positions=32,
    )
    expected = np.linspace(0, len(encoded.full_bpe_ids) - 1, 30, dtype=int)
    np.testing.assert_array_equal(encoded.retained_full_bpe_indices, expected)
    assert not encoded.fully_retained


def test_instrumented_token_ids_equal_upstream(cico_tokenizer) -> None:
    text = "We're going to work on graceful hand movements in front of you."
    encoded = tokenize_with_spans(
        text,
        text_uid="h2:train:fixture",
        tokenizer=cico_tokenizer,
        max_positions=32,
    )
    upstream = [cico_tokenizer.encoder["<|startoftext|>"], *cico_tokenizer.encode(text)]
    if len(upstream) > 31:
        indexes = [0, *np.linspace(1, len(upstream) - 1, 30, dtype=int).tolist()]
        upstream = [upstream[index] for index in indexes]
    upstream.append(cico_tokenizer.encoder["<|endoftext|>"])
    upstream.extend([0] * (32 - len(upstream)))
    assert encoded.input_ids == tuple(upstream)
