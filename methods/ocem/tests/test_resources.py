from __future__ import annotations

import hashlib

from ocem.provenance.resources import build_resource_lock, verify_resource


def test_file_hash_and_delimited_content_pass(tmp_path) -> None:
    path = tmp_path / "annotations.tsv"
    path.write_text("VIDEO_ID\tSENTENCE\na\tfirst\nb\tsecond\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = verify_resource(
        {
            "id": "annotations",
            "kind": "annotation",
            "required": True,
            "local_path": str(path),
            "expected_sha256": digest,
            "expected_bytes": path.stat().st_size,
            "access_status": "present",
            "excluded_dependency": False,
            "loader": {
                "kind": "delimited_text",
                "delimiter": "\t",
                "expected_rows": 2,
                "header_contains": ["VIDEO_ID", "SENTENCE"],
            },
        }
    )
    assert result["status"] == "PASS"
    assert result["content"]["rows"] == 2


def test_hash_mismatch_fails_before_content_load(tmp_path) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"not a model")
    result = verify_resource(
        {
            "id": "model",
            "kind": "checkpoint",
            "required": True,
            "local_path": str(path),
            "expected_sha256": "0" * 64,
            "access_status": "present",
            "excluded_dependency": False,
            "loader": {"kind": "torch_checkpoint"},
        }
    )
    assert result["status"] == "FAIL_HASH"
    assert result["loaded"] is False


def test_excluded_dependency_is_never_resolved() -> None:
    result = verify_resource(
        {
            "id": "seds_artifacts",
            "kind": "excluded_dependency",
            "required": False,
            "local_path": "/must/not/be/read",
            "access_status": "excluded",
            "excluded_dependency": True,
        }
    )
    assert result == {
        "id": "seds_artifacts",
        "kind": "excluded_dependency",
        "required": False,
        "excluded_dependency": True,
        "source_url": None,
        "configured_access_status": "excluded",
        "status": "PASS_EXCLUDED",
        "local_path": None,
        "loaded": False,
    }


def test_resource_lock_does_not_promote_scientific_gate(tmp_path) -> None:
    directory = tmp_path / "dataset"
    directory.mkdir()
    config = {
        "schema_version": "ocem.resources.v1",
        "resources": [
            {
                "id": "dataset",
                "kind": "dataset_directory",
                "required": True,
                "local_path": str(directory),
                "access_status": "present",
                "excluded_dependency": False,
                "loader": {"kind": "directory"},
            }
        ],
    }
    lock = build_resource_lock(config, tmp_path / "resources.yaml")
    assert lock["status"] == "PASS"
    assert lock["scientific_gate_promoted"] is False
    assert lock["seds_dependency"] is False

