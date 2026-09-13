from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from method1.config import load_config
from method1.manifests import build_manifests, iter_jsonl


def test_grouped_manifest_builder_preserves_membership(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SLRET_DATA_ROOT", str(tmp_path / "placeholder"))
    monkeypatch.setenv("SLRET_RUN_ROOT", str(tmp_path / "runs"))
    base = load_config("methods/sssc/configs/method1/ph_span_shared.yaml")
    annotation_paths = {}
    memberships = {}
    agnostic_root = tmp_path / "agnostic"
    aware_root = tmp_path / "aware"
    for split in ("train", "dev", "test"):
        group_id = f"group-{split}"
        memberships[split] = [group_id]
        annotation_path = tmp_path / f"{split}.pkl"
        records = [
            {"video_name": f"video-{split}-a", "text": f"Caption {split}", "ori_text": ""},
            {"video_name": f"video-{split}-b", "text": f"Caption {split}", "ori_text": ""},
        ]
        with annotation_path.open("wb") as handle:
            pickle.dump({group_id: records}, handle)
        annotation_paths[split] = str(annotation_path)
        for root, offset in ((agnostic_root, 0.0), (aware_root, 1.0)):
            (root / split).mkdir(parents=True)
            for record in records:
                with (root / split / f"{record['video_name']}.pkl").open("wb") as handle:
                    pickle.dump(
                        {
                            "name": record["video_name"],
                            "feature": np.full((2, 1024), offset, dtype=np.float32),
                        },
                        handle,
                    )
    membership_path = tmp_path / "membership.json"
    membership_path.write_text(json.dumps(memberships), encoding="utf-8")
    data = replace(
        base.data,
        dataset="h2",
        source_annotations=annotation_paths,
        official_membership_json=str(membership_path),
        official_split_annotations={},
        manifest_dir=str(tmp_path / "manifests"),
        agnostic_root=str(agnostic_root),
        aware_root=str(aware_root),
        agnostic_weight=0.8,
    )
    config = replace(base, data=data)
    config.validate()
    report = build_manifests(config)
    assert report["status"] == "ready"
    assert report["files"]["texts.jsonl"]["count"] == 3
    assert report["files"]["groups.jsonl"]["count"] == 3
    assert report["files"]["videos.jsonl"]["count"] == 6
    groups = list(iter_jsonl(tmp_path / "manifests" / "groups.jsonl"))
    assert [len(group["video_uids"]) for group in groups] == [2, 2, 2]
