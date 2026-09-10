"""Implementation-state validation and checkpoint rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ocem.provenance.gates import GateStatus


class StateError(ValueError):
    """Raised when implementation state is incomplete or internally invalid."""


_REQUIRED = {
    "schema_version",
    "spec_version",
    "research_checkpoint",
    "research_decision",
    "method",
    "seds_dependency",
    "current_work_package",
    "completed_work_packages",
    "gates",
    "open_blockers",
    "next_action",
}


def validate_implementation_state(state: Mapping[str, Any]) -> None:
    missing = sorted(_REQUIRED - set(state))
    if missing:
        raise StateError(f"implementation state missing keys: {', '.join(missing)}")
    if state["schema_version"] != "ocem.implementation_state.v1":
        raise StateError("unsupported implementation state schema")
    if state["method"] != "OCEM":
        raise StateError("implementation state method must be OCEM")
    if state["seds_dependency"] is not False:
        raise StateError("OCEM implementation state must keep seds_dependency=false")
    if not isinstance(state["gates"], Mapping):
        raise StateError("gates must be a mapping")
    for name, raw_status in state["gates"].items():
        try:
            GateStatus(raw_status)
        except ValueError as error:
            raise StateError(f"gate {name!r} has invalid status {raw_status!r}") from error


def load_implementation_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StateError(f"cannot load implementation state {state_path}: {error}") from error
    if not isinstance(state, dict):
        raise StateError("implementation state must be a JSON object")
    validate_implementation_state(state)
    return state


def render_checkpoint(state: Mapping[str, Any]) -> str:
    validate_implementation_state(state)
    completed = ", ".join(state["completed_work_packages"]) or "None"
    blockers = state["open_blockers"]
    blocker_lines = "\n".join(f"- {item}" for item in blockers) if blockers else "- None"
    gate_lines = "\n".join(f"- `{name}`: `{status}`" for name, status in state["gates"].items())
    return (
        "# OCEM Implementation Checkpoint\n\n"
        f"- Research checkpoint: {state['research_checkpoint']}\n"
        f"- Research decision: {state['research_decision']}\n"
        f"- Current work package: {state['current_work_package']}\n"
        f"- Completed work packages: {completed}\n"
        "- Neural retrieval result claimed: No\n\n"
        "## Gates\n\n"
        f"{gate_lines}\n\n"
        "## Open blockers\n\n"
        f"{blocker_lines}\n\n"
        "## Exact next action\n\n"
        f"{state['next_action']}\n\n"
        "## Resume rule\n\n"
        f"Continue from this state and Checkpoint {state['research_checkpoint']}. Do not restart "
        "broad literature search, "
        "change OCEM's core method, add SEDS/Baidu dependencies, or promote NOT_RUN gates to PASS "
        "without their required artifacts.\n"
    )
