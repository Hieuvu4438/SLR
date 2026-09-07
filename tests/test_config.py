from __future__ import annotations

from pathlib import Path

import pytest

from elsc.config import ConfigError, config_hash, load_config


def test_recursive_merge_and_stable_hash(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("schema_version: 1\ntrain: {eval_split: dev, epochs: 20}\nvalues: [1, 2]\n")
    child.write_text("extends: base.yaml\ntrain: {epochs: 5}\nvalues: [3]\n")
    result = load_config(child, validate=False)
    assert result["train"] == {"eval_split": "dev", "epochs": 5}
    assert result["values"] == [3]
    assert config_hash(result) == config_hash(load_config(child, validate=False))


def test_cycle_is_rejected(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nschema_version: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nschema_version: 1\n")
    with pytest.raises(ConfigError, match="cyclic"):
        load_config(tmp_path / "a.yaml", validate=False)


def test_test_selection_and_filtered_gallery_are_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nmethod: baseline\ntrain: {eval_split: test}\n"
        "evaluation: {filter_by_aux_eligibility: true}\n"
    )
    with pytest.raises(ConfigError, match="dev split"):
        load_config(path)


def test_unknown_precision_is_rejected(tmp_path: Path):
    path = tmp_path / "bad_precision.yaml"
    path.write_text(
        "schema_version: 1\nmethod: baseline\ntrain: {eval_split: dev, precision: int8}\n"
        "model: {backbone_frozen: false}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="train.precision"):
        load_config(path)


def test_full_rejects_unimplemented_intervention_fill(tmp_path: Path):
    path = tmp_path / "bad_fill.yaml"
    path.write_text(
        "schema_version: 1\nmethod: elsc\n"
        "train: {eval_split: dev}\n"
        "model: {backbone_frozen: true, adapter: {enabled: true}}\n"
        "sources: {temporal_metadata_root: /rf}\n"
        "evidence: {enabled: true, control_same_token_count: true, mask_fill: learned}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="mask_fill=zero"):
        load_config(path)


@pytest.mark.parametrize("field", ["encoder_microbatch_size", "score_microbatch_size"])
def test_full_rejects_nonpositive_microbatch_size(tmp_path: Path, field: str):
    path = tmp_path / "bad_evidence_microbatch.yaml"
    path.write_text(
        "schema_version: 1\nmethod: elsc\n"
        "train: {eval_split: dev}\n"
        "model: {backbone_frozen: true, adapter: {enabled: true}}\n"
        "sources: {temporal_metadata_root: /rf}\n"
        "evidence: {enabled: true, control_same_token_count: true, "
        f"mask_fill: zero, {field}: 0}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_nonzero_adapter_radius_is_rejected_until_dense_path_exists(tmp_path: Path):
    path = tmp_path / "bad_radius.yaml"
    path.write_text(
        "schema_version: 1\nmethod: elsc\n"
        "train: {eval_split: dev}\n"
        "model: {backbone_frozen: true, adapter: {enabled: true, radius: 1}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="radius=0"):
        load_config(path)


def test_nonpositive_epoch_count_is_rejected(tmp_path: Path):
    path = tmp_path / "zero_epochs.yaml"
    path.write_text(
        "schema_version: 1\nmethod: baseline\n"
        "train: {eval_split: dev, epochs: 0}\n"
        "model: {backbone_frozen: false}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="epochs"):
        load_config(path)


def test_required_feature_sidecars_must_pin_expected_provenance(tmp_path: Path):
    path = tmp_path / "weak_sidecar.yaml"
    path.write_text(
        "schema_version: 1\nmethod: baseline\n"
        "train: {eval_split: dev}\n"
        "model: {backbone_frozen: false}\n"
        "sources: {require_feature_sidecars: true}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="expected feature provenance"):
        load_config(path)


def test_random_span_duration_tolerance_is_bounded(tmp_path: Path):
    path = tmp_path / "bad_random_span.yaml"
    path.write_text(
        "schema_version: 1\nmethod: baseline\n"
        "train: {eval_split: dev}\n"
        "model: {backbone_frozen: false}\n"
        "cache: {random_span_duration_tolerance: 1.1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="random_span_duration_tolerance"):
        load_config(path)
