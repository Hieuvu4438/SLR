from __future__ import annotations

import csv
import pickle
import subprocess
from pathlib import Path

from elsc.audit import audit_assets


def test_asset_audit_can_lock_test_out_of_transfer_preparation(tmp_path):
    pair_ids = {"train": "train-video", "dev": "dev-video"}
    sources: dict[str, object] = {
        "prepared_splits": ["train", "dev"],
        "official_annotation_delimiter": ",",
        "test_annotation": None,
        "test_official_annotation": None,
    }
    for split, pair_id in pair_ids.items():
        annotation = tmp_path / f"{split}.pkl"
        with annotation.open("wb") as handle:
            pickle.dump({pair_id: {"video_name": pair_id, "text": split}}, handle)
        official = tmp_path / f"{split}.csv"
        with official.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("name", "text"))
            writer.writeheader()
            writer.writerow({"name": pair_id, "text": split})
        sources[f"{split}_annotation"] = str(annotation)
        sources[f"{split}_official_annotation"] = str(official)
    for stream in ("agnostic", "aware"):
        root = tmp_path / stream
        for split, pair_id in pair_ids.items():
            directory = root / split
            directory.mkdir(parents=True)
            (directory / f"{pair_id}.pkl").write_bytes(b"feature")
        sources[f"feature_{stream}_root"] = str(root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    project = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = {
        "schema_version": 1,
        "method": "baseline",
        "sources": sources,
        "data": {"dataset": "csl_daily"},
        "model": {"init_checkpoint": str(checkpoint), "teacher_checkpoint": None},
        "upstream": {"cico_root": str(project), "cico_commit": commit},
        "resources": {"min_free_disk_gib": 0},
    }
    result = audit_assets(config, tmp_path / "configs" / "csl_base.yaml")
    assert result["status"] == "ready"
    assert result["diagnostics"]["prepared_splits"] == ["train", "dev"]
    assert result["paths"]["sources.test_annotation"]["status"] == "unresolved"
