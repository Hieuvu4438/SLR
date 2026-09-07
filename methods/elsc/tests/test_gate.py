from __future__ import annotations

import json

import pytest

from elsc.config import config_hash
from elsc.data.manifest import ManifestRecord
from elsc.data.views import canonical_view
from elsc.gate import (
    GateContractError,
    evaluate_full_gate,
    evaluate_gain_gate,
    evaluate_mechanism_gate,
    main,
    validate_rf_cache_contract,
)


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


@pytest.mark.parametrize(
    ("gain", "expected_status", "expected_exit_code"),
    [(0.6, "passed", 0), (0.4, "no_go", 1)],
)
def test_gate_cli_exit_code_tracks_gate_status(tmp_path, gain, expected_status, expected_exit_code):
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "gate.json"
    report_path.write_text(json.dumps(_report(gain, gain)), encoding="utf-8")

    exit_code = main(
        [
            "--gate",
            "G",
            "--report",
            str(report_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == expected_exit_code
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == expected_status


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
                "training_control_contract_hash": f"matched-{seed}",
            }
            for seed in seeds
        ],
        "baseline_runs": [
            {
                "seed": seed,
                "training_control_contract_hash": f"matched-{seed}",
            }
            for seed in seeds
        ],
    }


def test_mechanism_gate_requires_true_support_to_win_both_controls_on_two_seeds():
    versus_random = _control_report([42, 1337, 2026], [0.4, 0.2, -0.1], [0.2, 0.4, 0.1])
    versus_caption = _control_report([42, 1337, 2026], [0.2, 0.6, 0.2], [0.4, 0.2, -0.4])
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

    mismatched_control = _control_report([42], [0.2], [0.4])
    mismatched_control["baseline_runs"][0]["training_control_contract_hash"] = "different"
    with pytest.raises(GateContractError, match="does not match"):
        evaluate_mechanism_gate(versus_random, mismatched_control)


def _passing_full_gate_inputs():
    gain = evaluate_gain_gate(_report(0.6, 0.8))
    random = _control_report([42, 1337], [0.4, 0.2], [0.2, 0.4])
    caption = _control_report([42, 1337], [0.2, 0.6], [0.4, 0.2])
    mechanism = evaluate_mechanism_gate(random, caption)
    config = {
        "model": {"teacher_checkpoint_sha256": "teacher"},
        "evidence": {
            "enabled": True,
            "require_verified_rf_metadata": True,
            "control_same_token_count": True,
            "control_duration_tolerance": 0.1,
            "teacher_margin_min": 0.02,
            "max_pairs_per_video": 1,
        },
    }
    records = [
        {
            "evidence_eligible": True,
            "video_id": "video-1",
            "evidence_remove_dense_indices": [1, 2],
            "control_remove_dense_indices": [5, 6],
            "intervention_coordinate_system": "input_frame",
            "teacher_clean_margin": 0.03,
            "evidence_interval": [10.0, 20.0],
            "control_interval": [30.0, 40.5],
        }
    ]
    cache = {
        "schema_version": 1,
        "split": "train",
        "config_hash": config_hash(config),
        "teacher_hash": "teacher",
        "manifest_hash": "manifest",
        "gates": {"evidence_eligible": 1},
    }
    rf_contract = {
        "criteria": {
            "manifest_pair_binding_pass": True,
            "verified_temporal_metadata_pass": True,
            "canonical_view_hash_pass": True,
            "evidence_rf_closure_pass": True,
            "control_rf_closure_pass": True,
            "other_support_exclusion_pass": True,
        },
        "observed": {"eligible_records_checked": 1},
    }
    return gain, mechanism, config, cache, records, rf_contract


def test_full_gate_requires_passed_dev_gates_and_valid_interventions():
    gain, mechanism, config, cache, records, rf_contract = _passing_full_gate_inputs()
    result = evaluate_full_gate(
        gain,
        mechanism,
        config,
        cache,
        records,
        cache_artifacts_match=True,
        train_manifest_sha256="manifest",
        rf_contract=rf_contract,
    )
    assert result["status"] == "passed"
    assert result["observed"]["eligible_evidence_records"] == 1

    gain["status"] = "no_go"
    result = evaluate_full_gate(
        gain,
        mechanism,
        config,
        cache,
        records,
        cache_artifacts_match=True,
        train_manifest_sha256="manifest",
        rf_contract=rf_contract,
    )
    assert result["status"] == "no_go"
    assert result["criteria"]["gain_gate_pass"] is False


def test_full_gate_rejects_dense_coordinates_and_overlapping_controls():
    gain, mechanism, config, cache, records, rf_contract = _passing_full_gate_inputs()
    records[0]["intervention_coordinate_system"] = "dense_index"
    records[0]["control_remove_dense_indices"] = [2, 6]
    result = evaluate_full_gate(
        gain,
        mechanism,
        config,
        cache,
        records,
        cache_artifacts_match=True,
        train_manifest_sha256="manifest",
        rf_contract=rf_contract,
    )
    assert result["status"] == "no_go"
    assert result["criteria"]["input_frame_coordinate_pass"] is False
    assert result["criteria"]["intervention_structure_pass"] is False


def test_full_rf_gate_rederives_masks_from_verified_temporal_sidecar(tmp_path):
    temporal_root = tmp_path / "temporal"
    temporal_root.mkdir()
    temporal_path = temporal_root / "video.json"
    temporal_path.write_text(
        json.dumps(
            {
                "verified": True,
                "coordinate_system": "input_frame",
                "recipe_sha256": "recipe",
                "rf_start": [0, 10, 20, 30, 40, 50],
                "rf_end": [10, 20, 30, 40, 50, 60],
            }
        ),
        encoding="utf-8",
    )
    manifest = ManifestRecord(
        schema_version=1,
        dataset="ph",
        split="train",
        pair_id="pair-1",
        video_id="video-1",
        caption_id="caption-1",
        caption_original="caption",
        caption_model="caption",
        caption_language="en",
        feature_agnostic="agnostic.pkl",
        feature_aware="aware.pkl",
        dense_length=6,
        feature_dim=1024,
        temporal_metadata=str(temporal_path),
    )
    view = canonical_view(6, 4)
    record = {
        "pair_id": "pair-1",
        "video_id": "video-1",
        "word_id": 1,
        "evidence_eligible": True,
        "view_hash": view.hash,
        "support_dense_indices": [1],
        "evidence_remove_dense_indices": [1],
        "control_remove_dense_indices": [3],
        "evidence_interval": [10, 20],
        "control_interval": [30, 40],
    }
    config = {
        "sources": {
            "temporal_metadata_root": str(temporal_root),
            "feature_recipe_sha256": "recipe",
        },
        "data": {"feature_len": 4},
    }

    result = validate_rf_cache_contract(config, [record], [manifest])
    assert all(result["criteria"].values())
    assert result["observed"]["eligible_records_checked"] == 1

    record["evidence_remove_dense_indices"] = [0]
    result = validate_rf_cache_contract(config, [record], [manifest])
    assert result["criteria"]["evidence_rf_closure_pass"] is False

    record["evidence_remove_dense_indices"] = [1]
    record["control_remove_dense_indices"] = [5]
    result = validate_rf_cache_contract(config, [record], [manifest])
    assert result["criteria"]["control_rf_closure_pass"] is False
