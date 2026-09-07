from __future__ import annotations

import pytest

from elsc.gate import GateContractError, evaluate_gain_gate


def _report(t2v: float, v2t: float, *, split: str = "dev"):
    return {
        "schema_version": 1,
        "result_kind": "measured_local",
        "split": split,
        "units": "percentage_points",
        "paired_seeds": [42],
        "delta": {
            "T2V": {"R1": {"mean": t2v}},
            "V2T": {"R1": {"mean": v2t}},
        },
    }


def test_gain_gate_requires_mean_gain_and_direction_safety():
    passed = evaluate_gain_gate(_report(0.6, 0.8))
    assert passed["status"] == "passed"
    assert passed["observed"]["mean_r1_gain_percentage_points"] == pytest.approx(0.7)

    weak_mean = evaluate_gain_gate(_report(0.4, 0.4))
    assert weak_mean["status"] == "no_go"
    assert weak_mean["criteria"]["mean_gain_pass"] is False

    unsafe_direction = evaluate_gain_gate(_report(-0.6, 2.0))
    assert unsafe_direction["status"] == "no_go"
    assert unsafe_direction["criteria"]["mean_gain_pass"] is True
    assert unsafe_direction["criteria"]["directions_pass"] is False


def test_gain_gate_rejects_test_feedback_and_nonmeasured_input():
    with pytest.raises(GateContractError, match="dev split"):
        evaluate_gain_gate(_report(1.0, 1.0, split="test"))
    invalid = _report(1.0, 1.0)
    invalid["result_kind"] = "placeholder"
    with pytest.raises(GateContractError, match="measured_local"):
        evaluate_gain_gate(invalid)
