from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from typing import Sequence

import regex


class TextUnitError(ValueError):
    """Text normalization, token offsets, or truncation violate the unit contract."""


NORMALIZATION_VERSION = "text_norm_v1"
UNITIZER_VERSION = "word_numeric_v1"

_UNIT_PATTERN = regex.compile(
    r"(?P<number>(?<![\p{L}\p{N}_])[+-]?(?:\d+(?:[.,]\d+)?)"
    r"(?:\s*(?:millimeters?|centimeters?|meters?|kilometers?|seconds?|minutes?|hours?|"
    r"grams?|kilograms?|degrees?|percent|%))?(?![\p{L}\p{N}_]))"
    r"|(?P<word>[\p{L}]+(?:['’-][\p{L}]+)*)",
    flags=regex.IGNORECASE,
)


@dataclass(frozen=True)
class TextUnit:
    index: int
    char_start: int
    char_end: int
    text: str
    normalized_value: str
    unit_kind: str


@dataclass(frozen=True)
class MappedTextUnit:
    unit: TextUnit
    subword_indices: tuple[int, ...]
    token_ids: tuple[int, ...]
    complete_after_truncation: bool


def normalize_text(text: str, version: str = NORMALIZATION_VERSION) -> str:
    if version != NORMALIZATION_VERSION:
        raise TextUnitError(f"unknown normalization version: {version}")
    if not isinstance(text, str):
        raise TextUnitError("text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def unitize(text_model: str, version: str = UNITIZER_VERSION) -> tuple[TextUnit, ...]:
    if version != UNITIZER_VERSION:
        raise TextUnitError(f"unknown unitizer version: {version}")
    units: list[TextUnit] = []
    for match in _UNIT_PATTERN.finditer(text_model):
        kind = "numeric_expression" if match.lastgroup == "number" else "word"
        raw = match.group(0)
        normalized = regex.sub(r"\s+", " ", raw.casefold()).strip()
        units.append(
            TextUnit(
                index=len(units),
                char_start=match.start(),
                char_end=match.end(),
                text=raw,
                normalized_value=normalized,
                unit_kind=kind,
            )
        )
    return tuple(units)


def map_units_to_subwords(
    units: Sequence[TextUnit],
    token_ids: Sequence[int],
    token_offsets: Sequence[tuple[int, int] | None],
) -> tuple[MappedTextUnit, ...]:
    """Map offsets emitted by the native tokenizer to unit spans.

    Special/padding tokens use a `None` offset. A truncated unit is retained for evidence auditing
    but marked incomplete and must not be selected as a local target.
    """
    if len(token_ids) != len(token_offsets):
        raise TextUnitError("token_ids and token_offsets must have the same length")
    previous_end = 0
    for offset in token_offsets:
        if offset is None:
            continue
        start, end = offset
        if start < 0 or end <= start or start < previous_end:
            raise TextUnitError("token offsets must be positive, ordered, half-open spans")
        previous_end = end
    mapped: list[MappedTextUnit] = []
    for unit in units:
        indices = tuple(
            index
            for index, offset in enumerate(token_offsets)
            if offset is not None and offset[0] < unit.char_end and offset[1] > unit.char_start
        )
        covered = [False] * (unit.char_end - unit.char_start)
        for index in indices:
            offset = token_offsets[index]
            assert offset is not None
            left = max(unit.char_start, offset[0])
            right = min(unit.char_end, offset[1])
            for position in range(left, right):
                covered[position - unit.char_start] = True
        mapped.append(
            MappedTextUnit(
                unit=unit,
                subword_indices=indices,
                token_ids=tuple(int(token_ids[index]) for index in indices),
                complete_after_truncation=bool(indices)
                and all(flag or char.isspace() for flag, char in zip(covered, unit.text, strict=True)),
            )
        )
    return tuple(mapped)


def require_target(mapping: Sequence[MappedTextUnit], unit_index: int) -> MappedTextUnit:
    if unit_index < 0 or unit_index >= len(mapping):
        raise TextUnitError(f"unit index out of range: {unit_index}")
    target = mapping[unit_index]
    if not target.complete_after_truncation:
        raise TextUnitError(f"unit {unit_index} is incomplete after truncation")
    return target


def unit_mapping_hash(mapping: Sequence[MappedTextUnit]) -> str:
    payload = [
        {
            **asdict(item.unit),
            "subword_indices": item.subword_indices,
            "token_ids": item.token_ids,
            "complete_after_truncation": item.complete_after_truncation,
        }
        for item in mapping
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
