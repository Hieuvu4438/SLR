from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from dive.cli import main
from dive.config import load_config
from dive.data.manifest import SampleRecord
from dive.data.relations import RelationError, load_excluded_negatives
from dive.data.relevance import RelevanceError, load_relevance
from dive.data.validation import DataValidationError, validate_prepared_data


HERE = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = HERE / "configs" / "fixture.yaml"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _record(split: str, index: int, *, source: str | None = None) -> SampleRecord:
    stem = f"{split}_{index}"
    return SampleRecord(
        schema_version="sample.v1",
        sample_id=f"sample_{stem}",
        video_id=f"video_{stem}",
        text_id=f"text_{stem}",
        split=split,
        sign_language="ase",
        text_language_original="en",
        text_language_model="en",
        text_original=f"caption {stem}",
        text_model=f"caption {stem}",
        source_video_id=source or f"source_{stem}",
        signer_id=None,
        source_start_sec=0.0,
        source_end_sec=1.0,
        duration_sec=1.0,
        video_path=f"{split}/{stem}.mp4",
        pose_path=f"{split}/{stem}.pkl",
        rgb_feature_key=f"rgb_{stem}",
        translation_artifact_hash=None,
        frame_map_key=f"frames_{stem}",
        annotation_provenance="unit_test",
    )


def _prepared_config(tmp_path: Path) -> tuple[dict[str, object], dict[str, list[SampleRecord]]]:
    config = load_config(FIXTURE_CONFIG)
    config["run"]["output_root"] = str(tmp_path / "runs")
    assets = tmp_path / "assets"
    records = {
        "train": [_record("train", 0), _record("train", 1)],
        "dev": [_record("dev", 0)],
        "test": [_record("test", 0)],
    }
    relevance_dir = tmp_path / "relevance"
    for split, split_records in records.items():
        manifest = tmp_path / "manifests" / f"{split}.jsonl"
        _write_jsonl(manifest, [asdict(record) for record in split_records])
        config["data"][f"{split}_manifest"] = str(manifest)
        _write_jsonl(
            relevance_dir / f"{split}.jsonl",
            [
                {
                    "schema_version": "relevance.v1",
                    "video_id": record.video_id,
                    "positive_text_ids": [record.text_id],
                }
                for record in split_records
            ],
        )
        for record in split_records:
            video = assets / str(record.video_path)
            pose = assets / str(record.pose_path)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
            pose.write_bytes(b"pose")
    relations = tmp_path / "relations" / "train_excluded_negatives.jsonl"
    _write_jsonl(
        relations,
        [
            {
                "schema_version": "excluded_negative.v1",
                "video_id": records["train"][0].video_id,
                "text_id": records["train"][1].text_id,
                "reason": "same_source_semantic_exclusion",
                "provenance": "unit_test",
            }
        ],
    )
    config["data"].update(
        {
            "video_root": str(assets),
            "pose_root": str(assets),
            "relevance_dir": str(relevance_dir),
            "train_relations": str(relations),
        }
    )
    return config, records


def test_relevance_requires_exact_bidirectional_coverage(tmp_path):
    path = tmp_path / "relevance.jsonl"
    _write_jsonl(
        path,
        [{"schema_version": "relevance.v1", "video_id": "v0", "positive_text_ids": ["t0"]}],
    )
    assert load_relevance(path, video_ids=["v0"], text_ids=["t0"]) == {
        "v0": frozenset({"t0"})
    }
    with pytest.raises(RelevanceError, match="no positive video"):
        load_relevance(path, video_ids=["v0"], text_ids=["t0", "t1"])


def test_excluded_negative_cannot_hide_a_positive(tmp_path):
    path = tmp_path / "relations.jsonl"
    _write_jsonl(
        path,
        [
            {
                "schema_version": "excluded_negative.v1",
                "video_id": "v0",
                "text_id": "t0",
                "reason": "bad",
                "provenance": "unit_test",
            }
        ],
    )
    with pytest.raises(RelationError, match="known positive"):
        load_excluded_negatives(
            path, video_ids=["v0"], text_ids=["t0"], positives_by_video={"v0": ["t0"]}
        )


def test_validate_data_cli_writes_and_registers_audit(tmp_path, capsys):
    config, _ = _prepared_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["validate-data", "--config", str(config_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ready"] is True
    assert printed["manifests"]["train"]["record_count"] == 2
    root = tmp_path / "runs" / "shared" / "seed17"
    audit = json.loads((root / "data_validation.json").read_text(encoding="utf-8"))
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    assert audit == printed
    assert state["stages"]["validate_data"]["outputs"]["audit"]["sha256"]


def test_validation_rejects_missing_required_asset(tmp_path):
    config, records = _prepared_config(tmp_path)
    broken = records["dev"][0]
    Path(config["data"]["pose_root"], str(broken.pose_path)).unlink()
    with pytest.raises(DataValidationError, match="MISSING_DATA_ASSET"):
        validate_prepared_data(config)


def test_validation_rejects_source_video_leakage(tmp_path):
    config, records = _prepared_config(tmp_path)
    dev = records["dev"][0]
    leaked = SampleRecord(**{**asdict(dev), "source_video_id": records["train"][0].source_video_id})
    _write_jsonl(Path(config["data"]["dev_manifest"]), [asdict(leaked)])
    with pytest.raises(DataValidationError, match="SPLIT_SOURCE_OVERLAP"):
        validate_prepared_data(config)
