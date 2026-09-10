from __future__ import annotations

import json

import pytest

from ocem.provenance.gates import GateError, require_gate
from ocem.provenance.hashes import canonical_json_sha256, sha256_file
from ocem.provenance.state import load_implementation_state, render_checkpoint


def test_hashes_are_deterministic(tmp_path) -> None:
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"ocem")
    assert sha256_file(path) == "e73e592264374ae214a663431c7e54c44b4a9c5244cdafc4dbde564169bef29f"
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256({"a": 1, "b": 2})


def test_gate_is_fail_closed() -> None:
    require_gate({"solver_gpu": "PASS"}, "solver_gpu")
    with pytest.raises(GateError, match="NOT_RUN"):
        require_gate({"solver_gpu": "NOT_RUN"}, "solver_gpu")
    with pytest.raises(GateError, match="absent"):
        require_gate({}, "solver_gpu")


def test_state_round_trip_and_checkpoint(tmp_path) -> None:
    state = {
        "schema_version": "ocem.implementation_state.v1",
        "spec_version": "1.0",
        "research_checkpoint": "8",
        "research_decision": "GO_WITH_CONDITIONS",
        "method": "OCEM",
        "seds_dependency": False,
        "current_work_package": "WP-01",
        "completed_work_packages": ["WP-00"],
        "gates": {"solver_gpu": "NOT_RUN"},
        "open_blockers": ["fixture blocker"],
        "next_action": "Continue implementation.",
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    loaded = load_implementation_state(path)
    checkpoint = render_checkpoint(loaded)
    assert "WP-01" in checkpoint
    assert "fixture blocker" in checkpoint
    assert "Neural retrieval result claimed: No" in checkpoint


def test_state_forbids_seds_dependency(tmp_path) -> None:
    state = {
        "schema_version": "ocem.implementation_state.v1",
        "spec_version": "1.0",
        "research_checkpoint": "8",
        "research_decision": "GO_WITH_CONDITIONS",
        "method": "OCEM",
        "seds_dependency": True,
        "current_work_package": "WP-01",
        "completed_work_packages": [],
        "gates": {},
        "open_blockers": [],
        "next_action": "Stop.",
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="seds_dependency=false"):
        load_implementation_state(path)
