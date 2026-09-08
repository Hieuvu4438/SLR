from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dive.artifacts import ArtifactResolver
from dive.cache import TensorShard, make_cache_fingerprint, write_tensor_cache
from dive.config import config_hash, load_config
from dive.data.manifest import SampleRecord
from dive.data.text_units import MappedTextUnit, unitize
from dive.mining.neighbors import NeighborProposal, _pair_id
from dive.mining.audit_runner import export_proposal_audit
from dive.mining.bank import load_bank
from dive.mining.finalize_runner import FinalizeRunError, finalize_train_contrasts
from dive.mining.runner import (
    _load_native_features,
    _semantic_candidates,
    build_forbidden_candidates,
    propose_train_contrasts,
)
from dive.mining.proposals import load_proposals


def _record(sample_id: str, video_id: str, text_id: str, text: str) -> SampleRecord:
    return SampleRecord(
        schema_version="sample.v1",
        sample_id=sample_id,
        video_id=video_id,
        text_id=text_id,
        split="train",
        sign_language="ASL",
        text_language_original="en",
        text_language_model="en",
        text_original=text,
        text_model=text,
        source_video_id=video_id,
        signer_id=None,
        source_start_sec=0.0,
        source_end_sec=1.0,
        duration_sec=1.0,
        video_path="video.mp4",
        pose_path="pose.pkl",
        rgb_feature_key=video_id,
        translation_artifact_hash=None,
        frame_map_key=sample_id,
        annotation_provenance="fixture",
    )


def _native_fingerprint(payload: str):
    return make_cache_fingerprint(
        "native_baseline",
        {
            "baseline_state": "baseline",
            "preprocessing": {"version": "fixture"},
            "tokenizer": "tokenizer",
            "sequence_selection": {"manifest": "manifest"},
            "masks": {"video": "bool", "text": "bool"},
            "sample_ids": {"namespace": "train"},
            "dtype": "float32",
            "payload": payload,
        },
    )


def test_forbidden_candidates_expand_shared_text_rows_and_exclusions():
    records = (
        _record("s0", "v0", "t0", "caption zero"),
        _record("s1", "v1", "t1", "caption one"),
        _record("s2", "v2", "t0", "caption zero"),
    )
    forbidden = build_forbidden_candidates(
        records,
        {"v0": {"t0"}, "v1": {"t1"}, "v2": {"t0"}},
        [("v0", "t1")],
    )
    assert forbidden[0] == {0, 1, 2}
    assert forbidden[1] == {1}
    assert forbidden[2] == {0, 2}


def test_native_cache_loader_binds_sample_and_text_order(tmp_path):
    records = (
        _record("s0", "v0", "t0", "caption zero"),
        _record("s1", "v1", "t1", "caption one"),
    )
    video_fingerprint = _native_fingerprint("video")
    text_fingerprint = _native_fingerprint("text")
    video_dir = tmp_path / "video"
    text_dir = tmp_path / "text"
    write_tensor_cache(
        video_dir,
        namespace="train_native_video",
        fingerprint=video_fingerprint,
        shards=(
            TensorShard(
                "00000",
                ("s0", "s1"),
                {
                    "pooled": torch.nn.functional.normalize(torch.randn(2, 4), dim=-1),
                    "fusion_hidden": torch.randn(2, 3, 4),
                },
                {"video": torch.tensor([[True, True, True], [True, True, False]])},
                {},
                {"split": "train", "contains_cls": True},
            ),
        ),
    )
    write_tensor_cache(
        text_dir,
        namespace="train_native_text",
        fingerprint=text_fingerprint,
        shards=(
            TensorShard(
                "00000",
                ("s0", "s1"),
                {
                    "pooled": torch.nn.functional.normalize(torch.randn(2, 4), dim=-1),
                    "tokens": torch.randn(2, 2, 4),
                },
                {"text": torch.tensor([[True, True], [True, False]])},
                {},
                {"split": "train", "text_ids": ["t0", "t1"]},
            ),
        ),
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "dive_frozen_train_cache.v1",
                "config_sha256": "config",
                "kind": "frozen_train",
                "test_content_used": False,
                "fingerprints": {
                    "native_video": video_fingerprint.to_dict(),
                    "native_text": text_fingerprint.to_dict(),
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = _load_native_features(
        SimpleNamespace(path=video_dir),
        SimpleNamespace(path=text_dir),
        SimpleNamespace(path=report_path),
        records,
        expected_config_hash="config",
    )
    assert loaded[0].shape == (2, 4)
    assert loaded[2].shape == (2, 3, 4)
    assert loaded[3].shape == (2, 2, 4)


def test_semantic_candidates_reject_incomplete_numeric_target():
    records = (
        _record("s0", "v0", "t0", "the value is 12 meters"),
        _record("s1", "v1", "t1", "the value is 15 meters"),
    )
    quartet = ((0.8, 0.9), (0.85, 0.8))
    proposal = NeighborProposal(
        pair_id=_pair_id("s0", "s1"),
        sample_i="s0",
        sample_j="s1",
        nominated_by=("s0",),
        directions=("v2t",),
        s0_quartet=quartet,
        margins0=(-0.1, -0.05, -0.05, -0.1),
        hardness=0.1,
    )

    def lineage(text: str, *, complete: bool):
        units = unitize(text)
        return SimpleNamespace(
            units=tuple(
                MappedTextUnit(
                    unit, (index,), (index,), complete or unit.unit_kind != "numeric_expression"
                )
                for index, unit in enumerate(units)
            )
        )

    accepted, rejected = _semantic_candidates(
        [proposal],
        records,
        {
            "t0": lineage(records[0].text_model, complete=True),
            "t1": lineage(records[1].text_model, complete=False),
        },
    )
    assert not accepted
    assert rejected == (
        {
            "schema_version": "semantic_rejection.v1",
            "reason": "target_unit_incomplete_after_truncation",
            "proposal": {
                "schema_version": "neighbor_proposal.v1",
                "pair_id": proposal.pair_id,
                "sample_i": "s0",
                "sample_j": "s1",
                "nominated_by": ("s0",),
                "directions": ("v2t",),
                "s0_quartet": quartet,
                "margins0": (-0.1, -0.05, -0.05, -0.1),
                "hardness": 0.1,
            },
        },
    )


def test_propose_runner_registers_train_only_outputs_and_is_idempotent(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "how2sign_base.yaml")
    config["run"]["output_root"] = str(tmp_path / "runs")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "video.mp4").write_bytes(b"fixture-video")
    config["data"]["video_root"] = str(video_root)
    config["mining"]["shortlist_per_direction"] = 2
    config["mining"]["rerank_topk"] = 1
    config["mining"]["shortlist_audit_queries"] = 2
    resolver = ArtifactResolver(config)
    data_dir = resolver.output_path("shared", "fixture_data")
    data_dir.mkdir(parents=True)
    records = tuple(
        _record(f"s{index}", f"v{index}", f"t{index}", f"the value is {10 + index} meters")
        for index in range(4)
    )
    manifest = data_dir / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    relations = data_dir / "relations.jsonl"
    relations.write_text("", encoding="utf-8")
    relevance_dir = data_dir / "relevance"
    relevance_dir.mkdir()
    (relevance_dir / "train.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "relevance.v1",
                    "video_id": record.video_id,
                    "positive_text_ids": [record.text_id],
                },
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    resolver.record_stage(
        "prepare_data",
        {
            "train_manifest": manifest,
            "train_relations": relations,
            "relevance_dir": relevance_dir,
        },
        scope="shared",
    )
    validation_dir = resolver.output_path("shared", "fixture_validation")
    unit_dir = validation_dir / "units"
    unit_dir.mkdir(parents=True)
    (unit_dir / "train.jsonl").write_text("fixture\n", encoding="utf-8")
    audit_path = validation_dir / "audit.json"
    audit_path.write_text("{}\n", encoding="utf-8")
    resolver.record_stage(
        "validate_data",
        {
            "audit": audit_path,
            "train_manifest": manifest,
            "train_relations": relations,
            "train_relevance": relevance_dir / "train.jsonl",
            "text_unit_maps_dir": unit_dir,
        },
        scope="shared",
    )

    video_fingerprint = _native_fingerprint("video")
    text_fingerprint = _native_fingerprint("text")
    cache_dir = resolver.output_path("shared", "fixture_cache")
    video_dir = cache_dir / "video"
    text_dir = cache_dir / "text"
    reference_dir = cache_dir / "reference"
    reference_dir.mkdir(parents=True)
    (reference_dir / "fixture.bin").write_bytes(b"reference-local-with-timestamps")
    generator = torch.Generator().manual_seed(233)
    pooled_video = torch.nn.functional.normalize(torch.randn(4, 5, generator=generator), dim=-1)
    pooled_text = torch.nn.functional.normalize(torch.randn(4, 5, generator=generator), dim=-1)
    write_tensor_cache(
        video_dir,
        namespace="train_native_video",
        fingerprint=video_fingerprint,
        shards=(
            TensorShard(
                "00000",
                tuple(item.sample_id for item in records),
                {
                    "pooled": pooled_video,
                    "fusion_hidden": torch.randn(4, 3, 5, generator=generator),
                },
                {"video": torch.ones(4, 3, dtype=torch.bool)},
                {},
                {"split": "train", "contains_cls": True},
            ),
        ),
    )
    write_tensor_cache(
        text_dir,
        namespace="train_native_text",
        fingerprint=text_fingerprint,
        shards=(
            TensorShard(
                "00000",
                tuple(item.sample_id for item in records),
                {
                    "pooled": pooled_text,
                    "tokens": torch.randn(4, 2, 5, generator=generator),
                },
                {"text": torch.ones(4, 2, dtype=torch.bool)},
                {},
                {"split": "train", "text_ids": [item.text_id for item in records]},
            ),
        ),
    )
    cache_report = cache_dir / "report.json"
    cache_report.write_text(
        json.dumps(
            {
                "schema_version": "dive_frozen_train_cache.v1",
                "config_sha256": config_hash(config),
                "kind": "frozen_train",
                "test_content_used": False,
                "implementation_git_revision": "cache-revision",
                "fingerprints": {
                    "native_video": video_fingerprint.to_dict(),
                    "native_text": text_fingerprint.to_dict(),
                },
            }
        ),
        encoding="utf-8",
    )
    resolver.record_stage(
        "cache_frozen_train",
        {
            "native_video": video_dir,
            "native_text": text_dir,
            "reference_local": reference_dir,
            "report": cache_report,
        },
        scope="shared",
    )

    def fake_lineages(_path, *, expected_texts):
        return {
            text_id: SimpleNamespace(
                unit_mapping_sha256=f"mapping-{text_id}",
                units=tuple(
                    MappedTextUnit(unit, (index,), (index,), True)
                    for index, unit in enumerate(unitize(text))
                ),
            )
            for text_id, text in expected_texts.items()
        }

    monkeypatch.setattr("dive.mining.runner.load_text_unit_lineage", fake_lineages)
    monkeypatch.setattr("dive.mining.runner._repository_revision", lambda _root: "a" * 40)
    monkeypatch.setattr("dive.mining.audit_runner._repository_revision", lambda _root: "a" * 40)
    monkeypatch.setattr("dive.mining.finalize_runner._repository_revision", lambda _root: "a" * 40)
    monkeypatch.setattr("dive.mining.finalize_runner.load_text_unit_lineage", fake_lineages)
    report = propose_train_contrasts(
        config,
        device="cpu",
        pooled_query_chunk_size=2,
        pair_chunk_size=2,
    )
    assert report["eligible_proposal_count"] > 0
    assert report["shortlist_coverage_total"] == 2
    proposal_parent = resolver.resolve("mine_propose", "proposals", scope="shared")
    proposals = load_proposals(
        proposal_parent.path,
        expected_fingerprint=report["mining_fingerprint"],
        expected_sample_ids=[item.sample_id for item in records],
    )
    assert len(proposals) == report["eligible_proposal_count"]
    state_path = resolver.state_path("shared")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"].pop("mine_propose")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    recovered = propose_train_contrasts(
        config,
        device="cpu",
        pooled_query_chunk_size=2,
        pair_chunk_size=2,
    )
    assert recovered["artifacts"]["report"]["sha256"]
    repeated = propose_train_contrasts(
        config,
        device="cpu",
        pooled_query_chunk_size=2,
        pair_chunk_size=2,
    )
    assert repeated == {key: value for key, value in recovered.items() if key != "artifacts"}
    audit = export_proposal_audit(config)
    assert audit["exported_size"] == min(2, report["eligible_proposal_count"])
    assert audit["student_outputs_included"] is False
    annotation_parent = resolver.resolve("audit_export", "annotation_template", scope="shared")
    annotation_rows = [
        json.loads(line) for line in annotation_parent.path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        row["video_i"] == str((video_root / "video.mp4").resolve()) for row in annotation_rows
    )
    assert all(row["rater_id"] is None and "student" not in row for row in annotation_rows)
    decision_parent = resolver.resolve("audit_export", "decision_template", scope="shared")
    decision = json.loads(decision_parent.path.read_text(encoding="utf-8"))
    assert decision["audit_status"] is None and decision["decision_author"] is None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"].pop("audit_export")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    audit_recovered = export_proposal_audit(config)
    assert audit_recovered["artifacts"]["annotation_template"]["sha256"]
    assert export_proposal_audit(config) == {
        key: value for key, value in audit_recovered.items() if key != "artifacts"
    }

    completed_annotation = tmp_path / "completed_annotation.jsonl"
    completed_rows = []
    for row in annotation_rows:
        row.update(
            {
                "category": "strict_numeric_length",
                "positive_i_rating": 5,
                "positive_j_rating": 5,
                "cross_i_j_negative_rating": 1,
                "cross_j_i_negative_rating": 1,
                "uncertain": False,
                "rater_id": "human-01",
            }
        )
        completed_rows.append(row)
    completed_annotation.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in completed_rows),
        encoding="utf-8",
    )
    decision.update(
        {
            "audit_id": "fixture-human-audit",
            "audit_status": "accepted",
            "annotation_path": str(completed_annotation),
            "annotation_sha256": hashlib.sha256(completed_annotation.read_bytes()).hexdigest(),
            "decision_author": "fixture-human",
            "decision_date_utc": "2026-09-08T12:00:00Z",
            "rationale": "fixture exercises the explicit human gate",
        }
    )
    completed_decision = tmp_path / "completed_decision.json"
    completed_decision.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")
    config["mining"]["schema_audit_artifact"] = str(completed_decision)

    dependency_dir = resolver.output_path("shared", "fixture_dependencies")
    dependency_dir.mkdir()
    baseline_report = dependency_dir / "baseline_report.json"
    baseline_report.write_text("{}\n", encoding="utf-8")
    reference = dependency_dir / "reference.pt"
    reference.write_bytes(b"reference")
    resolver.record_stage("baseline_validate", {"report": baseline_report}, scope="shared")
    resolver.record_stage("evidence_warmup", {"reference": reference}, scope="shared")
    finalized = finalize_train_contrasts(config)
    assert finalized["audit_status"] == "accepted"
    assert finalized["bank_record_count"] == audit["proposal_count"]
    bank_parent = resolver.resolve("mine_finalize", "pre_support_bank", scope="shared")
    assert (
        len(load_bank(bank_parent.path, finalized["bank_fingerprint"]))
        == finalized["bank_record_count"]
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"].pop("mine_finalize")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    finalized_recovered = finalize_train_contrasts(config)
    assert finalized_recovered["artifacts"]["pre_support_bank"]["sha256"]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"].pop("mine_finalize")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    rejection_path = (
        Path(config["run"]["output_root"])
        / "shared"
        / "seed17"
        / "mining"
        / "finalized"
        / "semantic_rejections.jsonl"
    )
    rejection_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FinalizeRunError, match="rejection artifact is invalid"):
        finalize_train_contrasts(config)
