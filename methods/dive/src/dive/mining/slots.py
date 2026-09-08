from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import regex

from dive.data.text_units import normalize_text, unitize


SCHEMA_VERSION = "strict_numeric_v1"

_AMBIGUOUS = regex.compile(
    r"\b(?:about|approximately|around|roughly|between|from|to|or|if|unless|not|no|"
    r"maybe|perhaps|range)\b|[~≈]",
    flags=regex.IGNORECASE,
)
_NUMBER = regex.compile(
    r"^(?P<value>[+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>[\p{L}%]+)$",
    flags=regex.IGNORECASE,
)
_UNITS = {
    "millimeter": "length:mm",
    "millimeters": "length:mm",
    "centimeter": "length:cm",
    "centimeters": "length:cm",
    "meter": "length:m",
    "meters": "length:m",
    "kilometer": "length:km",
    "kilometers": "length:km",
    "second": "duration:s",
    "seconds": "duration:s",
    "minute": "duration:min",
    "minutes": "duration:min",
    "hour": "duration:h",
    "hours": "duration:h",
    "gram": "mass:g",
    "grams": "mass:g",
    "kilogram": "mass:kg",
    "kilograms": "mass:kg",
    "degree": "angle:degree",
    "degrees": "angle:degree",
    "percent": "ratio:percent",
    "%": "ratio:percent",
}


@dataclass(frozen=True)
class SchemaAudit:
    schema_id: str
    audit_status: str
    artifact_hash: str | None
    audit_id: str

    def validate(self) -> None:
        if self.schema_id != SCHEMA_VERSION:
            raise ValueError(f"audit schema must be {SCHEMA_VERSION}")
        if self.audit_status not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"invalid audit status: {self.audit_status}")
        if self.audit_status == "accepted" and not self.artifact_hash:
            raise ValueError("accepted schema requires a real audit artifact hash")


@dataclass(frozen=True)
class AtomicSlotResult:
    eligible: bool
    unit_i: int | None
    unit_j: int | None
    category: str | None
    g_sem: float
    reason: str
    schema_version: str
    audit_id: str


def _reject(reason: str, audit: SchemaAudit) -> AtomicSlotResult:
    return AtomicSlotResult(False, None, None, None, 0.0, reason, SCHEMA_VERSION, audit.audit_id)


def validate_strict_numeric_pair(
    text_i: str,
    text_j: str,
    audit: SchemaAudit,
) -> AtomicSlotResult:
    """Validate one unambiguous measurement slot changed between otherwise equal captions."""
    audit.validate()
    first = normalize_text(text_i)
    second = normalize_text(text_j)
    if _AMBIGUOUS.search(first) or _AMBIGUOUS.search(second):
        return _reject("ambiguous_scope_or_approximation", audit)
    units_i = unitize(first)
    units_j = unitize(second)
    if len(units_i) != len(units_j):
        return _reject("unit_count_mismatch", audit)
    differences = [
        index
        for index, (left, right) in enumerate(zip(units_i, units_j, strict=True))
        if left.normalized_value != right.normalized_value
    ]
    if len(differences) != 1:
        return _reject("requires_exactly_one_changed_unit", audit)
    index = differences[0]
    left = units_i[index]
    right = units_j[index]
    if left.unit_kind != "numeric_expression" or right.unit_kind != "numeric_expression":
        return _reject("changed_unit_is_not_numeric", audit)
    left_match = _NUMBER.fullmatch(left.normalized_value)
    right_match = _NUMBER.fullmatch(right.normalized_value)
    if left_match is None or right_match is None:
        return _reject("numeric_expression_requires_explicit_unit", audit)
    unit_i = _UNITS.get(left_match.group("unit").casefold())
    unit_j = _UNITS.get(right_match.group("unit").casefold())
    if unit_i is None or unit_j is None or unit_i != unit_j:
        return _reject("numeric_type_or_unit_mismatch", audit)
    try:
        value_i = Decimal(left_match.group("value").replace(",", "."))
        value_j = Decimal(right_match.group("value").replace(",", "."))
    except InvalidOperation:
        return _reject("invalid_numeric_value", audit)
    if not value_i.is_finite() or not value_j.is_finite() or value_i == value_j:
        return _reject("numeric_values_not_distinct", audit)
    if audit.audit_status != "accepted":
        return AtomicSlotResult(
            True,
            index,
            index,
            f"strict_numeric_{unit_i.split(':', 1)[0]}",
            0.0,
            f"schema_{audit.audit_status}",
            SCHEMA_VERSION,
            audit.audit_id,
        )
    return AtomicSlotResult(
        True,
        index,
        index,
        f"strict_numeric_{unit_i.split(':', 1)[0]}",
        1.0,
        "accepted",
        SCHEMA_VERSION,
        audit.audit_id,
    )
