from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ocem.data.extraction import ExtractionRunError, extract_p14t_i3d_features
from ocem.provenance.hashes import sha256_file


def _fixture(tmp_path: Path) -> dict:
    manifest_dir = tmp_path / "manifests"
    video_root = tmp_path / "videos"
    train_root = video_root / "train"
    shared_root = tmp_path / "shared"
    for path in (manifest_dir, train_root, shared_root / "slr_common"):
        path.mkdir(parents=True, exist_ok=True)
    video = train_root / "sample-a.mp4"
    video.write_bytes(b"video")
    (manifest_dir / "train.jsonl").write_text(
        json.dumps({"sample_id": "sample-a", "raw_relpath": "train/sample-a.mp4"}) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "adapted.pt"
    extractor = tmp_path / "i3d.py"
    upstream = tmp_path / "upstream_i3d.py"
    checkpoint.write_bytes(b"checkpoint")
    extractor.write_text("# extractor\n", encoding="utf-8")
    upstream.write_text("# model\n", encoding="utf-8")
    return {
        "manifest_dir": manifest_dir,
        "video_root": video_root,
        "checkpoint": checkpoint,
        "expected_checkpoint_sha256": sha256_file(checkpoint),
        "output_root": tmp_path / "features",
        "temporal_root": tmp_path / "temporal",
        "shared_root": shared_root,
        "extractor": extractor,
        "expected_extractor_sha256": sha256_file(extractor),
        "upstream_i3d": upstream,
        "expected_upstream_i3d_sha256": sha256_file(upstream),
        "splits": ["train"],
    }


def test_wrapper_generates_manifest_list_and_checks_shared_report(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path)

    def fake_run(command, *, env, check):
        assert check is False
        assert str(Path(inputs["shared_root"]).resolve()) in env["PYTHONPATH"]
        output_root = Path(command[command.index("--output-root") + 1])
        split = command[command.index("--splits") + 1]
        output_root.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "planned",
            "videos": 1,
            "windows": 3,
            "completed": 0,
            "resumed": 0,
            "checkpoint_sha256": inputs["expected_checkpoint_sha256"],
        }
        (output_root / f"extraction_report_{split}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("ocem.data.extraction.subprocess.run", fake_run)
    report = extract_p14t_i3d_features(**inputs, dry_run=True)
    assert report["status"] == "PLANNED"
    assert report["manifests"]["train"]["samples"] == 1
    video_list = Path(report["manifests"]["train"]["video_list"])
    assert video_list.read_text(encoding="utf-8") == "sample-a.mp4\n"
    assert "--dry-run" in report["commands"][0]


def test_wrapper_rejects_directory_ids_outside_manifest(tmp_path, monkeypatch) -> None:
    inputs = _fixture(tmp_path)
    (Path(inputs["video_root"]) / "train" / "extra.mp4").write_bytes(b"extra")
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("ocem.data.extraction.subprocess.run", fake_run)
    with pytest.raises(ExtractionRunError, match="differ from manifest"):
        extract_p14t_i3d_features(**inputs, dry_run=True)
    assert called is False
