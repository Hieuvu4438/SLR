from __future__ import annotations

import pytest

from elsc.gate import GateContractError, evaluate_gain_gate, evaluate_mechanism_gate


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


def _control_report(seeds, t2v, v2t, *, method_hash="selected-min"):
    return {
        "schema_version": 1,
        "result_kind": "measured_local",
        "split": "dev",
        "units": "percentage_points",
        "paired_seeds": seeds,
        "test_used_for_tuning": None,
        "delta": {
            "T2V": {"R1": {"values": t2v}},
            "V2T": {"R1": {"values": v2t}},
        },
        "method_runs": [
            {
                "seed": seed,
                "config_hash": method_hash,
                "selection_sha256": f"selection-{seed}",
                "checkpoint_sha256": f"checkpoint-{seed}",
                "metrics_sha256": f"metrics-{seed}",
            }
            for seed in seeds
        ],
    }


def test_mechanism_gate_requires_true_support_to_win_both_controls_on_two_seeds():
    versus_random = _control_report(
        [42, 1337, 2026], [0.4, 0.2, -0.1], [0.2, 0.4, 0.1]
    )
    versus_caption = _control_report(
        [42, 1337, 2026], [0.2, 0.6, 0.2], [0.4, 0.2, -0.4]
    )
    result = evaluate_mechanism_gate(versus_random, versus_caption)
    assert result["status"] == "passed"
    assert result["observed"]["winning_seeds"] == [42, 1337]

    versus_caption["delta"]["V2T"]["R1"]["values"][1] = -1.0
    result = evaluate_mechanism_gate(versus_random, versus_caption)
    assert result["status"] == "no_go"
    assert result["observed"]["winning_seeds"] == [42]


def test_mechanism_gate_reports_insufficient_seeds_and_checks_pairing():
    versus_random = _control_report([42], [0.4], [0.2])
    versus_caption = _control_report([42], [0.2], [0.4])
    result = evaluate_mechanism_gate(versus_random, versus_caption)
    assert result["status"] == "insufficient_seeds"
    assert result["observed"]["winning_seed_count"] == 1

    mismatched = _control_report([42], [0.2], [0.4], method_hash="other-min")
    with pytest.raises(GateContractError, match="same true-support runs"):
        evaluate_mechanism_gate(versus_random, mismatched)

    mismatched = _control_report([1337], [0.2], [0.4])
    with pytest.raises(GateContractError, match="identical paired seeds"):
        evaluate_mechanism_gate(versus_random, mismatched)
