"""Fail-closed prerequisite gate semantics."""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL_TECHNICAL = "FAIL_TECHNICAL"
    BLOCKED_RESOURCE = "BLOCKED_RESOURCE"
    NO_GO_SCIENTIFIC = "NO_GO_SCIENTIFIC"
    NOT_RUN = "NOT_RUN"


class GateError(RuntimeError):
    """Raised when an operation lacks a passing prerequisite gate."""


def require_gate(gates: Mapping[str, str], gate_name: str) -> None:
    raw_status = gates.get(gate_name)
    if raw_status is None:
        raise GateError(f"required gate {gate_name!r} is absent")
    try:
        status = GateStatus(raw_status)
    except ValueError as error:
        raise GateError(f"required gate {gate_name!r} has invalid status {raw_status!r}") from error
    if status is not GateStatus.PASS:
        raise GateError(f"required gate {gate_name!r} is {status.value}, not PASS")

