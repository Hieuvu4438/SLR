from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from ocem.data.adaptation import build_p14t_adaptation_plan
from ocem.data.pseudolabels import (
    PseudoLabelError,
    generate_p14t_pseudolabel_index,
    passes_confidence_threshold,
    select_cico_segments,
)
from ocem.provenance.hashes import canonical_json_sha256, sha256_file


def test_cico_grouping_preserves_observed_nonlocal_merge_and_slice_boundary() -> None:
    segments = select_cico_segments(
        [
            {"start": 0, "end": 16, "confidence": 0.9},
            {"start": 30, "end": 46, "confidence": 0.8},
            {"start": 32, "end": 48, "confidence": 0.7},
            {"start": 12, "end": 28, "confidence": 0.6},
            {"start": 60, "end": 76, "confidence": 0.5},
        ]
    )
    assert [item["anchor_start"] for item in segments] == [0, 30, 60]
    # Start 32 is close to anchor 30, but upstream merges it into first anchor 0.
    assert segments[0]["merged_window_starts"] == [0, 32]
    assert segments[0]["support_end_frame_exclusive"] == 48
    assert segments[0]["upstream_clip_end_frame_exclusive"] == 47
    # Start 12 is suppressed by the <=24 rule and does not become a segment.
    assert all(item["anchor_start"] != 12 for item in segments)


def test_confidence_threshold_is_strict_and_rejects_nonfinite_values() -> None:
    assert not passes_confidence_threshold(0.6)
    assert passes_confidence_threshold(float(np.nextafter(0.6, 1.0)))
    assert not passes_confidence_threshold(float("nan"))


def _manifest(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    raw = tmp_path / "raw" / "train"
    manifests = tmp_path / "manifests"
    features = tmp_path / "features"
    temporal = tmp_path / "temporal"
    for path in (raw, manifests, features, temporal):
        path.mkdir(parents=True, exist_ok=True)
    train_records = []
    for sample_id in ("train-a", "train-b"):
        video = raw / f"{sample_id}.mp4"
        video.write_bytes(b"video")
        train_records.append({"sample_id": sample_id, "raw_relpath": f"raw/train/{sample_id}.mp4"})
    train = manifests / "train.jsonl"
    validation = manifests / "validation.jsonl"
    test = manifests / "test.jsonl"
    _manifest(train, train_records)
    _manifest(validation, [{"sample_id": "validation-a"}])
    _manifest(test, [{"sample_id": "test-a"}])
    vocabulary = tmp_path / "class.txt"
    vocabulary.write_text("0 zero\n1 one\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    state = {
        "module.logits.conv3d.weight": torch.zeros((2, 1024, 1, 1, 1)),
        "module.logits.conv3d.bias": torch.tensor([2.0, 0.0]),
    }
    torch.save({"state_dict": state}, checkpoint)
    plan = build_p14t_adaptation_plan(
        train_manifest=train,
        forbidden_manifests=[validation, test],
        checkpoint=checkpoint,
        expected_checkpoint_sha256=sha256_file(checkpoint),
        class_vocabulary=vocabulary,
        holdout_modulus=2,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for record in train_records:
        sample_id = record["sample_id"]
        feature_path = features / f"{sample_id}.pkl"
        source_video = tmp_path / record["raw_relpath"]
        with feature_path.open("wb") as handle:
            pickle.dump(
                {
                    "name": str(source_video),
                    "feature": np.zeros((2, 1024), dtype=np.float32),
                },
                handle,
                protocol=5,
            )
        metadata = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "feature_sha256": sha256_file(feature_path),
            "source_video": str(source_video),
            "stream_name": "domain_agnostic",
        }
        Path(f"{feature_path}.meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        support = {
            "coordinate_system": "input_frame",
            "decoded_frame_count": 17,
            "interval_convention": "half_open",
            "rf_start": [0, 1],
            "rf_end": [16, 17],
            "source_video": str(source_video),
        }
        (temporal / f"{sample_id}.json").write_text(json.dumps(support), encoding="utf-8")
    return {
        "adaptation_plan": plan_path,
        "expected_plan_sha256": sha256_file(plan_path),
        "feature_dir": features,
        "temporal_dir": temporal,
        "checkpoint": checkpoint,
        "expected_checkpoint_sha256": sha256_file(checkpoint),
        "output_index": tmp_path / "pseudo.jsonl",
    }


def test_generator_is_train_only_deterministic_and_uses_strict_threshold(tmp_path) -> None:
    inputs = _fixture(tmp_path)
    first = generate_p14t_pseudolabel_index(**inputs, device="cpu", batch_windows=1)
    first_bytes = Path(inputs["output_index"]).read_bytes()
    second = generate_p14t_pseudolabel_index(**inputs, device="cpu", batch_windows=2)
    assert Path(inputs["output_index"]).read_bytes() == first_bytes
    assert first["output"]["index_sha256"] == second["output"]["index_sha256"]
    assert first["provenance_checks"]["validation_test_overlap_count"] == 0
    records = [json.loads(line) for line in first_bytes.decode().splitlines()]
    assert {record["source_sample_id"] for record in records} == {"train-a", "train-b"}
    # softmax([2, 0]) > 0.6; the adjacent windows merge into one segment per sample.
    assert len(records) == 2
    assert all(record["merged_window_count"] == 2 for record in records)


def test_generator_rejects_feature_inventory_or_plan_drift(tmp_path) -> None:
    inputs = _fixture(tmp_path)
    extra = Path(inputs["feature_dir"]) / "validation-a.pkl"
    extra.write_bytes(b"forbidden")
    with pytest.raises(PseudoLabelError, match="features IDs do not exactly equal"):
        generate_p14t_pseudolabel_index(**inputs, device="cpu")
    extra.unlink()
    plan_path = Path(inputs["adaptation_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["validation_or_test_used"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    inputs["expected_plan_sha256"] = sha256_file(plan_path)
    with pytest.raises(PseudoLabelError, match="validation_or_test_used"):
        generate_p14t_pseudolabel_index(**inputs, device="cpu")


def test_records_digest_is_content_based(tmp_path) -> None:
    inputs = _fixture(tmp_path)
    report = generate_p14t_pseudolabel_index(**inputs, device="cpu")
    records = [json.loads(line) for line in Path(inputs["output_index"]).read_text().splitlines()]
    assert report["output"]["records_sha256"] == canonical_json_sha256(records)
