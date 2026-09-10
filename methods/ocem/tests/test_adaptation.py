from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocem.data.adaptation import AdaptationPlanError, build_p14t_adaptation_plan
from ocem.provenance.hashes import sha256_file


def _manifest(path: Path, ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in ids),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _manifest(train, [f"train-{index}" for index in range(100)])
    _manifest(validation, ["validation-1"])
    _manifest(test, ["test-1"])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    vocabulary = tmp_path / "class.txt"
    vocabulary.write_text("0 alpha\n1 beta\n", encoding="utf-8")
    return train, validation, test, checkpoint, vocabulary


def _build(tmp_path: Path):
    train, validation, test, checkpoint, vocabulary = _fixture(tmp_path)
    return build_p14t_adaptation_plan(
        train_manifest=train,
        forbidden_manifests=[validation, test],
        checkpoint=checkpoint,
        expected_checkpoint_sha256=sha256_file(checkpoint),
        class_vocabulary=vocabulary,
    )


def test_plan_is_train_only_complete_and_deterministic(tmp_path) -> None:
    first = _build(tmp_path)
    second = _build(tmp_path)
    assert first == second
    assert first["source_manifest"]["sample_count"] == 100
    assert first["forbidden_overlap_count"] == 0
    assert first["validation_or_test_used"] is False
    counts = first["split_policy"]["train_count"] + first["split_policy"]["holdout_count"]
    assert counts == 100
    assert {item["sample_id"] for item in first["assignments"]} == {
        f"train-{index}" for index in range(100)
    }


def test_plan_rejects_forbidden_id_overlap(tmp_path) -> None:
    train, validation, test, checkpoint, vocabulary = _fixture(tmp_path)
    _manifest(validation, ["train-7"])
    with pytest.raises(AdaptationPlanError, match="overlaps forbidden IDs"):
        build_p14t_adaptation_plan(
            train_manifest=train,
            forbidden_manifests=[validation, test],
            checkpoint=checkpoint,
            expected_checkpoint_sha256=sha256_file(checkpoint),
            class_vocabulary=vocabulary,
        )


def test_plan_rejects_checkpoint_or_vocabulary_drift(tmp_path) -> None:
    train, validation, test, checkpoint, vocabulary = _fixture(tmp_path)
    with pytest.raises(AdaptationPlanError, match="checkpoint SHA-256 mismatch"):
        build_p14t_adaptation_plan(
            train_manifest=train,
            forbidden_manifests=[validation, test],
            checkpoint=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            class_vocabulary=vocabulary,
        )
    vocabulary.write_text("1 noncontiguous\n", encoding="utf-8")
    with pytest.raises(AdaptationPlanError, match="contiguous labels"):
        build_p14t_adaptation_plan(
            train_manifest=train,
            forbidden_manifests=[validation, test],
            checkpoint=checkpoint,
            expected_checkpoint_sha256=sha256_file(checkpoint),
            class_vocabulary=vocabulary,
        )
