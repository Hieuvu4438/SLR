from __future__ import annotations

import hashlib
import html
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import regex
import ftfy


class TextUnitError(ValueError):
    """Text normalization, token offsets, or truncation violate the unit contract."""


NORMALIZATION_VERSION = "text_norm_v1"
SEDS_CLIP_NORMALIZATION_VERSION = "seds_clip_text_norm_v1"
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


@dataclass(frozen=True)
class TextUnitLineage:
    text_id: str
    text_model: str
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int] | None, ...]
    units: tuple[MappedTextUnit, ...]
    unit_mapping_sha256: str


def normalize_text(text: str, version: str = NORMALIZATION_VERSION) -> str:
    if not isinstance(text, str):
        raise TextUnitError("text must be a string")
    if version == NORMALIZATION_VERSION:
        return " ".join(unicodedata.normalize("NFC", text).split())
    if version == SEDS_CLIP_NORMALIZATION_VERSION:
        # Exact text entering the pinned CLIP regex/BPE path: basic_clean,
        # whitespace_clean, then lowercase in SimpleTokenizer.tokenize().
        cleaned = ftfy.fix_text(text)
        cleaned = html.unescape(html.unescape(cleaned))
        return regex.sub(r"\s+", " ", cleaned).strip().lower()
    raise TextUnitError(f"unknown normalization version: {version}")


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
    previous_start = 0
    for offset in token_offsets:
        if offset is None:
            continue
        start, end = offset
        # Byte-level BPE can split one Unicode code point, in which case two pieces
        # legitimately share the same covering character span.
        if start < 0 or end <= start or start < previous_start:
            raise TextUnitError("token offsets must be positive, ordered, half-open spans")
        previous_start = start
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
                and all(
                    flag or char.isspace() for flag, char in zip(covered, unit.text, strict=True)
                ),
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


def _strict_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise TextUnitError(f"{context} has invalid fields: {', '.join(details)}")


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TextUnitError(f"{context} must be an integer")
    return value


def _parse_stored_unit(raw: Any, context: str) -> MappedTextUnit:
    if not isinstance(raw, Mapping):
        raise TextUnitError(f"{context} must be an object")
    _strict_keys(
        raw,
        {"unit", "subword_indices", "token_ids", "complete_after_truncation"},
        context,
    )
    unit_raw = raw["unit"]
    if not isinstance(unit_raw, Mapping):
        raise TextUnitError(f"{context}.unit must be an object")
    _strict_keys(
        unit_raw,
        {"index", "char_start", "char_end", "text", "normalized_value", "unit_kind"},
        f"{context}.unit",
    )
    for name in ("text", "normalized_value", "unit_kind"):
        if not isinstance(unit_raw[name], str) or not unit_raw[name]:
            raise TextUnitError(f"{context}.unit.{name} must be a nonempty string")
    unit = TextUnit(
        index=_integer(unit_raw["index"], f"{context}.unit.index"),
        char_start=_integer(unit_raw["char_start"], f"{context}.unit.char_start"),
        char_end=_integer(unit_raw["char_end"], f"{context}.unit.char_end"),
        text=unit_raw["text"],
        normalized_value=unit_raw["normalized_value"],
        unit_kind=unit_raw["unit_kind"],
    )
    indices_raw = raw["subword_indices"]
    token_ids_raw = raw["token_ids"]
    if not isinstance(indices_raw, list) or not isinstance(token_ids_raw, list):
        raise TextUnitError(f"{context} token fields must be JSON lists")
    complete = raw["complete_after_truncation"]
    if not isinstance(complete, bool):
        raise TextUnitError(f"{context}.complete_after_truncation must be bool")
    return MappedTextUnit(
        unit=unit,
        subword_indices=tuple(
            _integer(value, f"{context}.subword_indices") for value in indices_raw
        ),
        token_ids=tuple(_integer(value, f"{context}.token_ids") for value in token_ids_raw),
        complete_after_truncation=complete,
    )


def _parse_lineage(raw: Mapping[str, Any], context: str) -> TextUnitLineage:
    _strict_keys(
        raw,
        {
            "schema_version",
            "text_id",
            "text_model",
            "token_ids",
            "token_offsets",
            "units",
            "unit_mapping_sha256",
        },
        context,
    )
    if raw["schema_version"] != "seds_text_unit_map.v1":
        raise TextUnitError(f"{context}.schema_version must be seds_text_unit_map.v1")
    for name in ("text_id", "text_model", "unit_mapping_sha256"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise TextUnitError(f"{context}.{name} must be a nonempty string")
    token_ids_raw = raw["token_ids"]
    offsets_raw = raw["token_offsets"]
    units_raw = raw["units"]
    if not isinstance(token_ids_raw, list) or not token_ids_raw:
        raise TextUnitError(f"{context}.token_ids must be a nonempty JSON list")
    if not isinstance(offsets_raw, list) or len(offsets_raw) != len(token_ids_raw):
        raise TextUnitError(f"{context}.token_offsets must align with token_ids")
    if not isinstance(units_raw, list):
        raise TextUnitError(f"{context}.units must be a JSON list")
    token_ids = tuple(_integer(value, f"{context}.token_ids") for value in token_ids_raw)
    if any(value < 0 for value in token_ids):
        raise TextUnitError(f"{context}.token_ids cannot contain negative values")
    offsets: list[tuple[int, int] | None] = []
    for index, value in enumerate(offsets_raw):
        if value is None:
            offsets.append(None)
            continue
        if not isinstance(value, list) or len(value) != 2:
            raise TextUnitError(f"{context}.token_offsets[{index}] must be null or [start,end]")
        offsets.append(
            (
                _integer(value[0], f"{context}.token_offsets[{index}][0]"),
                _integer(value[1], f"{context}.token_offsets[{index}][1]"),
            )
        )
    text_length = len(raw["text_model"])
    if any(
        offset is not None and (offset[0] < 0 or offset[1] <= offset[0] or offset[1] > text_length)
        for offset in offsets
    ):
        raise TextUnitError(f"{context}.token_offsets escape text_model")
    units = tuple(
        _parse_stored_unit(value, f"{context}.units[{index}]")
        for index, value in enumerate(units_raw)
    )
    expected_units = map_units_to_subwords(unitize(raw["text_model"]), token_ids, offsets)
    if len(units) != len(expected_units):
        raise TextUnitError(f"{context}.units do not derive from the stored native token lineage")
    for stored, expected in zip(units, expected_units, strict=True):
        if (
            stored.unit != expected.unit
            or stored.subword_indices != expected.subword_indices
            or stored.token_ids != expected.token_ids
            or (stored.complete_after_truncation and not expected.complete_after_truncation)
        ):
            raise TextUnitError(
                f"{context}.units do not derive from the stored native token lineage"
            )
    mapping_sha256 = unit_mapping_hash(units)
    if raw["unit_mapping_sha256"] != mapping_sha256:
        raise TextUnitError(f"{context}.unit_mapping_sha256 does not match its units")
    return TextUnitLineage(
        text_id=raw["text_id"],
        text_model=raw["text_model"],
        token_ids=token_ids,
        token_offsets=tuple(offsets),
        units=units,
        unit_mapping_sha256=mapping_sha256,
    )


def load_text_unit_lineage(
    path: str | Path,
    *,
    expected_texts: Mapping[str, str] | None = None,
) -> dict[str, TextUnitLineage]:
    """Load a checksummed validation-stage unit map and verify its native derivation."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TextUnitError(f"text-unit lineage artifact does not exist: {source}")
    records: dict[str, TextUnitLineage] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        context = f"{source}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TextUnitError(f"invalid JSON at {context}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise TextUnitError(f"{context} must contain an object")
        record = _parse_lineage(raw, context)
        if record.text_id in records:
            raise TextUnitError(f"duplicate text ID in unit lineage: {record.text_id}")
        records[record.text_id] = record
    if not records:
        raise TextUnitError(f"text-unit lineage artifact is empty: {source}")
    if expected_texts is not None:
        expected = {str(key): str(value) for key, value in expected_texts.items()}
        if set(records) != set(expected):
            missing = sorted(set(expected) - set(records))
            unknown = sorted(set(records) - set(expected))
            raise TextUnitError(
                f"text-unit lineage coverage mismatch: missing={missing}, unknown={unknown}"
            )
        mismatched = sorted(
            text_id
            for text_id, text_model in expected.items()
            if records[text_id].text_model != text_model
        )
        if mismatched:
            raise TextUnitError(f"text-unit lineage text mismatch: {mismatched}")
    return records
