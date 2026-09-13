from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

import ftfy
import numpy as np
import regex

from .utils import sha256_text


_HYPHENS = "-‐‑‒–—"


def basic_clean(text: str) -> str:
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def whitespace_clean(text: str) -> str:
    return regex.sub(r"\s+", " ", text).strip()


def canonicalize(text: str) -> str:
    return whitespace_clean(basic_clean(text)).lower()


@dataclass(frozen=True)
class LexicalSpan:
    occurrence_uid: str
    surface: str
    char_span: tuple[int, int]
    full_bpe_indices: tuple[int, ...]
    token_positions: tuple[int, ...]
    fully_retained: bool


@dataclass(frozen=True)
class TokenizedCaption:
    text_uid: str
    raw_text: str
    canonical_text: str
    caption_hash: str
    full_bpe_tokens: tuple[str, ...]
    full_bpe_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    text_valid: tuple[bool, ...]
    retained_full_bpe_indices: tuple[int, ...]
    inverse_retained_map: tuple[int, ...]
    lexical_spans: tuple[LexicalSpan, ...]
    fully_retained: bool


def _match_is_complete_lexical_unit(canonical: str, start: int, end: int, surface: str) -> bool:
    if not surface or not all(character.isalpha() for character in surface):
        return False
    before = canonical[start - 1] if start else ""
    after = canonical[end] if end < len(canonical) else ""
    # The CLIP regex splits contractions and hyphenated compounds into multiple matches.
    # Version 1 excludes every component of those constructs.
    if (before and before in "'’" + _HYPHENS) or (after and after in "'’" + _HYPHENS):
        return False
    return True


def tokenize_with_spans(
    raw_text: str,
    *,
    text_uid: str,
    tokenizer: Any,
    max_positions: int = 32,
) -> TokenizedCaption:
    if max_positions < 2:
        raise ValueError("max_positions must leave room for BOS and EOT")
    canonical = canonicalize(raw_text)
    pieces: list[str] = []
    matches: list[tuple[str, tuple[int, int], tuple[int, ...]]] = []
    for match in tokenizer.pat.finditer(canonical):
        surface = match.group(0)
        encoded = "".join(tokenizer.byte_encoder[value] for value in surface.encode("utf-8"))
        match_pieces = tokenizer.bpe(encoded).split(" ")
        first = len(pieces)
        pieces.extend(match_pieces)
        matches.append((surface, (int(match.start()), int(match.end())), tuple(range(first, len(pieces)))))

    upstream_tokens = tuple(tokenizer.tokenize(raw_text))
    if tuple(pieces) != upstream_tokens:
        raise ValueError("instrumented tokenizer diverged from pinned SimpleTokenizer")
    full_ids = tuple(int(value) for value in tokenizer.convert_tokens_to_ids(pieces))
    capacity = max_positions - 2
    if len(pieces) <= capacity:
        retained = tuple(range(len(pieces)))
    else:
        retained = tuple(int(value) for value in np.linspace(0, len(pieces) - 1, capacity, dtype=int))
    if len(set(retained)) != len(retained):
        raise ValueError("baseline linspace policy produced duplicate retained positions")

    bos = int(tokenizer.encoder["<|startoftext|>"])
    eot = int(tokenizer.encoder["<|endoftext|>"])
    selected_ids = [full_ids[index] for index in retained]
    ids = [bos, *selected_ids, eot]
    valid = [True] * len(ids)
    ids.extend([0] * (max_positions - len(ids)))
    valid.extend([False] * (max_positions - len(valid)))

    # Independent replication of the released loader's one-based selection logic.
    upstream_words = ["<|startoftext|>", *upstream_tokens]
    if len(upstream_words) > max_positions - 1:
        indexes = [
            0,
            *np.linspace(1, len(upstream_words) - 1, max_positions - 2, dtype=int).tolist(),
        ]
        upstream_words = [upstream_words[index] for index in indexes]
    upstream_words.append("<|endoftext|>")
    expected_ids = [int(value) for value in tokenizer.convert_tokens_to_ids(upstream_words)]
    expected_ids.extend([0] * (max_positions - len(expected_ids)))
    if tuple(ids) != tuple(expected_ids):
        raise AssertionError("token IDs do not match released UPRet loader behavior")

    inverse = [-1] * len(full_ids)
    for encoded_position, full_index in enumerate(retained, start=1):
        inverse[full_index] = encoded_position
    spans: list[LexicalSpan] = []
    for surface, (start, end), full_indices in matches:
        if not _match_is_complete_lexical_unit(canonical, start, end, surface):
            continue
        retained_complete = all(inverse[index] >= 0 for index in full_indices)
        positions = tuple(inverse[index] for index in full_indices if inverse[index] >= 0)
        spans.append(
            LexicalSpan(
                occurrence_uid=f"{text_uid}:{start}:{end}",
                surface=surface,
                char_span=(start, end),
                full_bpe_indices=full_indices,
                token_positions=positions,
                fully_retained=retained_complete,
            )
        )
    return TokenizedCaption(
        text_uid=text_uid,
        raw_text=raw_text,
        canonical_text=canonical,
        caption_hash=sha256_text(canonical),
        full_bpe_tokens=tuple(pieces),
        full_bpe_ids=full_ids,
        input_ids=tuple(ids),
        text_valid=tuple(valid),
        retained_full_bpe_indices=retained,
        inverse_retained_map=tuple(inverse),
        lexical_spans=tuple(spans),
        fully_retained=len(pieces) <= capacity,
    )


def replace_occurrence(
    original: TokenizedCaption,
    occurrence: LexicalSpan,
    replacement: str,
    *,
    tokenizer: Any,
    max_positions: int = 32,
) -> tuple[TokenizedCaption, LexicalSpan]:
    replacement = canonicalize(replacement)
    if not replacement or not all(character.isalpha() for character in replacement):
        raise ValueError("version 1 replacement must be one alphabetic lexical unit")
    if replacement == occurrence.surface:
        raise ValueError("replacement must differ from source word")
    start, end = occurrence.char_span
    if original.canonical_text[start:end] != occurrence.surface:
        raise ValueError("occurrence span does not index its source surface")
    negative_text = original.canonical_text[:start] + replacement + original.canonical_text[end:]
    negative = tokenize_with_spans(
        negative_text,
        text_uid=original.text_uid,
        tokenizer=tokenizer,
        max_positions=max_positions,
    )
    expected_span = (start, start + len(replacement))
    candidates = [span for span in negative.lexical_spans if span.char_span == expected_span]
    if len(candidates) != 1 or candidates[0].surface != replacement:
        raise ValueError("replacement did not round-trip to one complete lexical span")
    return negative, candidates[0]


def auxiliary_edit_eligible(
    positive: TokenizedCaption,
    positive_span: LexicalSpan,
    negative: TokenizedCaption,
    negative_span: LexicalSpan,
) -> tuple[bool, str | None]:
    if not positive.fully_retained:
        return False, "positive_caption_overlength"
    if not negative.fully_retained:
        return False, "negative_caption_overlength"
    if not positive_span.fully_retained or not negative_span.fully_retained:
        return False, "edited_span_not_fully_retained"
    if not positive_span.token_positions or not negative_span.token_positions:
        return False, "empty_edited_span"
    return True, None
