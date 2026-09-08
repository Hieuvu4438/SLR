from __future__ import annotations

import json
from pathlib import Path

import pytest

from dive.artifacts import ArtifactError, ArtifactResolver, hash_artifact
from dive.config import load_config
from dive.doctor import inspect_environment


HERE = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = HERE / "configs" / "fixture.yaml"


def _resolver(tmp_path: Path) -> ArtifactResolver:
    config = load_config(FIXTURE_CONFIG)
    return ArtifactResolver(config, output_root=tmp_path / "runs")


def test_run_layout_separates_shared_and_variant_outputs(tmp_path):
    resolver = _resolver(tmp_path)
    assert resolver.shared_root == (tmp_path / "runs/shared/seed17").resolve()
    assert resolver.experiment_root == (
        tmp_path / "runs/primary_pair/A6/seed17"
    ).resolve()
    assert resolver.output_path("shared", "baseline", "locked.pt") == (
        tmp_path / "runs/shared/seed17/baseline/locked.pt"
    ).resolve()
    with pytest.raises(ArtifactError, match="safe"):
        resolver.output_path("shared", "..", "escaped.pt")
    with pytest.raises(ArtifactError, match="escapes"):
        resolver.owned_path("shared", tmp_path / "outside.pt")


def test_stage_record_is_atomic_idempotent_and_resolvable(tmp_path):
    resolver = _resolver(tmp_path)
    artifact = resolver.output_path("shared", "baseline.json")
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"metric": 0.5}\n', encoding="utf-8")
    records = resolver.record_stage(
        "baseline_validate",
        {"report": artifact},
        scope="shared",
        metadata={"split": "dev"},
    )
    assert resolver.record_stage(
        "baseline_validate",
        {"report": artifact},
        scope="shared",
        metadata={"split": "dev"},
    ) == records
    resolved = resolver.resolve("baseline_validate", "report", scope="shared")
    assert resolved.path == artifact
    assert resolved.record.sha256 == hash_artifact(artifact)[0]
    state = json.loads(resolver.state_path("shared").read_text(encoding="utf-8"))
    assert state["schema_version"] == "dive_run_state.v1"
    assert "variant" not in state["layout"]
    assert state["stages"]["baseline_validate"]["config_sha256"] == resolver.config_sha256


def test_mutation_and_conflicting_reregistration_fail_closed(tmp_path):
    resolver = _resolver(tmp_path)
    artifact = resolver.output_path("experiment", "metrics.json")
    artifact.parent.mkdir(parents=True)
    artifact.write_text("first\n", encoding="utf-8")
    resolver.record_stage("evaluate_dev", {"metrics": artifact}, scope="experiment")
    artifact.write_text("second\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="CACHE_HASH_MISMATCH"):
        resolver.resolve("evaluate_dev", "metrics", scope="experiment")
    with pytest.raises(ArtifactError, match="different provenance"):
        resolver.record_stage("evaluate_dev", {"metrics": artifact}, scope="experiment")


def test_parent_resolution_never_discovers_unregistered_files(tmp_path):
    resolver = _resolver(tmp_path)
    arbitrary = resolver.output_path("shared", "found_checkpoint.pt")
    arbitrary.parent.mkdir(parents=True)
    arbitrary.write_bytes(b"not a registered checkpoint")
    with pytest.raises(ArtifactError, match="MISSING_PARENT_ARTIFACT"):
        resolver.resolve("baseline_train", "locked_checkpoint", scope="shared")


def test_parent_is_revalidated_when_recording_child(tmp_path):
    resolver = _resolver(tmp_path)
    parent_path = resolver.output_path("shared", "reference.pt")
    parent_path.parent.mkdir(parents=True)
    parent_path.write_bytes(b"reference")
    resolver.record_stage("evidence_warmup", {"reference": parent_path}, scope="shared")
    parent = resolver.resolve("evidence_warmup", "reference", scope="shared")
    child_path = resolver.output_path("experiment", "selection.json")
    child_path.parent.mkdir(parents=True)
    child_path.write_text("{}\n", encoding="utf-8")
    records = resolver.record_stage(
        "train",
        {"selection": child_path},
        scope="experiment",
        parents=[parent],
    )
    assert records["selection"].kind == "file"
    parent_path.write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="CACHE_HASH_MISMATCH"):
        resolver.record_stage(
            "calibrate",
            {"selection": child_path},
            scope="experiment",
            parents=[parent],
        )


def test_directory_hash_is_stable_and_rejects_symlinks(tmp_path):
    directory = tmp_path / "tree"
    directory.mkdir()
    (directory / "b.txt").write_text("b", encoding="utf-8")
    (directory / "a.txt").write_text("a", encoding="utf-8")
    first = hash_artifact(directory)
    assert first[1:] == (2, 2, "directory")
    assert hash_artifact(directory) == first
    (directory / "link").symlink_to(directory / "a.txt")
    with pytest.raises(ArtifactError, match="symlink"):
        hash_artifact(directory)


def test_artifact_path_must_be_owned_by_its_scope(tmp_path):
    resolver = _resolver(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="escapes"):
        resolver.record_stage("prepare_data", {"audit": outside}, scope="shared")


def test_doctor_resolves_and_rehashes_null_prepared_parent_paths(tmp_path):
    config = load_config(FIXTURE_CONFIG)
    config["run"]["output_root"] = str(tmp_path / "runs")
    resolver = ArtifactResolver(config)
    manifest = resolver.output_path("shared", "data", "train.jsonl")
    manifest.parent.mkdir(parents=True)
    manifest.write_text("prepared\n", encoding="utf-8")
    resolver.record_stage("prepare_data", {"train_manifest": manifest}, scope="shared")

    report = inspect_environment(config, "baseline_train")
    train = next(item for item in report["resources"] if item["field"] == "data.train_manifest")
    assert train == {
        "field": "data.train_manifest",
        "path": str(manifest),
        "exists": True,
        "error_code": None,
        "resolution": "run_state:shared/prepare_data/train_manifest",
        "detail": None,
    }

    manifest.write_text("mutated\n", encoding="utf-8")
    report = inspect_environment(config, "baseline_train")
    train = next(item for item in report["resources"] if item["field"] == "data.train_manifest")
    assert train["exists"] is False
    assert train["error_code"] == "CACHE_HASH_MISMATCH"
    assert "registered artifact changed" in train["detail"]
