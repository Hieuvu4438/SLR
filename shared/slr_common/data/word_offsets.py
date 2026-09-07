from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class WordUnit:
    surface: str
    canonical_char_span: tuple[int, int]
    full_bpe_indices: tuple[int, ...]
    full_bpe_ids: tuple[int, ...]
    encoded_bpe_positions: tuple[int, ...]
    fully_retained: bool
    occurrence_count: int


@dataclass(frozen=True)
class EncodedCaption:
    canonical: str
    token_ids: tuple[int, ...]
    input_mask: tuple[int, ...]
    selected_full_bpe_indices: tuple[int, ...]
    words: tuple[WordUnit, ...]
    was_subsampled: bool


def canonicalize(text: str, *, basic_clean: Callable[[str], str], whitespace_clean: Callable[[str], str]) -> str:
    return whitespace_clean(basic_clean(text)).lower()


def _is_alpha(surface: str) -> bool:
    return bool(surface) and all(character.isalpha() for character in surface)


def encode_with_offsets(
    text: str,
    tokenizer: Any,
    max_words: int,
    *,
    basic_clean: Callable[[str], str],
    whitespace_clean: Callable[[str], str],
    eligible: Callable[[str], bool] | None = None,
) -> EncodedCaption:
    """Mirror CiCo tokenization/subsampling and preserve canonical word-to-BPE identity."""
    if max_words < 2:
        raise ValueError("max_words must leave room for BOS and EOS")
    canonical = canonicalize(text, basic_clean=basic_clean, whitespace_clean=whitespace_clean)
    pieces: list[str] = []
    matches: list[tuple[str, tuple[int, int], tuple[int, ...]]] = []
    for match in tokenizer.pat.finditer(canonical):
        surface = match.group(0)
        byte_token = "".join(tokenizer.byte_encoder[value] for value in surface.encode("utf-8"))
        word_pieces = tokenizer.bpe(byte_token).split(" ")
        start = len(pieces)
        pieces.extend(word_pieces)
        matches.append((surface, match.span(), tuple(range(start, len(pieces)))))

    expected_pieces = list(tokenizer.tokenize(text))
    if pieces != expected_pieces:
        raise ValueError("offset tokenizer diverged from upstream tokenizer.tokenize")

    capacity = max_words - 2
    if len(pieces) > capacity:
        selected = np.linspace(0, len(pieces) - 1, capacity, dtype=int).tolist() if capacity else []
        was_subsampled = True
    else:
        selected = list(range(len(pieces)))
        was_subsampled = False
    selected_set = set(selected)
    bos = tokenizer.encoder["<|startoftext|>"]
    eos = tokenizer.encoder["<|endoftext|>"]
    selected_pieces = [pieces[index] for index in selected]
    token_ids = [bos, *tokenizer.convert_tokens_to_ids(selected_pieces), eos]
    input_mask = [1] * len(token_ids)
    token_ids.extend([0] * (max_words - len(token_ids)))
    input_mask.extend([0] * (max_words - len(input_mask)))

    # Independent reproduction of the exact upstream loader algorithm.
    upstream_words = ["<|startoftext|>", *expected_pieces]
    if len(upstream_words) > max_words - 1:
        indexes = [0, *np.linspace(1, len(upstream_words) - 1, max_words - 2, dtype=int).tolist()]
        upstream_words = [upstream_words[index] for index in indexes]
    upstream_words.append("<|endoftext|>")
    upstream_ids = tokenizer.convert_tokens_to_ids(upstream_words)
    upstream_ids.extend([0] * (max_words - len(upstream_ids)))
    if token_ids != upstream_ids:
        raise AssertionError("encoded token IDs do not match CiCo loader behavior")

    position_by_full_index = {full_index: position + 1 for position, full_index in enumerate(selected)}
    occurrence_counts: dict[str, int] = {}
    for surface, _, _ in matches:
        occurrence_counts[surface] = occurrence_counts.get(surface, 0) + 1
    predicate = eligible or _is_alpha
    words: list[WordUnit] = []
    piece_ids = tokenizer.convert_tokens_to_ids(pieces)
    for surface, span, full_indices in matches:
        if not predicate(surface):
            continue
        retained = all(index in selected_set for index in full_indices)
        words.append(
            WordUnit(
                surface=surface,
                canonical_char_span=(int(span[0]), int(span[1])),
                full_bpe_indices=full_indices,
                full_bpe_ids=tuple(piece_ids[index] for index in full_indices),
                encoded_bpe_positions=tuple(position_by_full_index[index] for index in full_indices if index in position_by_full_index),
                fully_retained=retained,
                occurrence_count=occurrence_counts[surface],
            )
        )
    return EncodedCaption(
        canonical=canonical,
        token_ids=tuple(token_ids),
        input_mask=tuple(input_mask),
        selected_full_bpe_indices=tuple(selected),
        words=tuple(words),
        was_subsampled=was_subsampled,
    )


def replace_canonical_span(canonical: str, span: tuple[int, int], replacement: str) -> str:
    start, end = span
    if not (0 <= start < end <= len(canonical)):
        raise ValueError(f"invalid canonical span {span} for length {len(canonical)}")
    replacement = replacement.strip().lower()
    if not replacement or any(character.isspace() for character in replacement):
        raise ValueError("MVP replacement must be one non-empty lexical unit")
    output = canonical[:start] + replacement + canonical[end:]
    if output == canonical:
        raise ValueError("replacement did not change the caption")
    return output


def eligible_word_units(encoded: EncodedCaption, stopwords: Iterable[str] = ()) -> list[WordUnit]:
    blocked = {item.lower() for item in stopwords}
    return [
        word
        for word in encoded.words
        if word.fully_retained and word.occurrence_count == 1 and word.surface not in blocked
    ]
