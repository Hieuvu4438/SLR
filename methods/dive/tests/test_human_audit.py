from __future__ import annotations

import hashlib
import json

import pytest

from dive.mining.audit import export_audit_template
from dive.mining.audit_runner import AuditRunError, _pair_ids_hash, load_semantic_audit_decision
from dive.mining.neighbors import NeighborProposal, _pair_id


def _proposal() -> NeighborProposal:
    return NeighborProposal(
        pair_id=_pair_id("s0", "s1"),
        sample_i="s0",
        sample_j="s1",
        nominated_by=("s0",),
        directions=("v2t",),
        s0_quartet=((0.8, 0.9), (0.85, 0.8)),
        margins0=(-0.1, -0.05, -0.05, -0.1),
        hardness=0.1,
    )


def _write_decision_bundle(tmp_path):
    template = tmp_path / "template.jsonl"
    exported = export_audit_template(
        [_proposal()],
        {"s0": "the value is 12 meters", "s1": "the value is 15 meters"},
        template,
        sample_size=1,
        seed=701,
        sample_context={
            "s0": {"video_path": "/videos/s0.mp4", "sign_language": "ASL", "text_language": "en"},
            "s1": {"video_path": "/videos/s1.mp4", "sign_language": "ASL", "text_language": "en"},
        },
    )
    report = tmp_path / "export_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "dive_audit_export.v1",
                "proposal_mining_fingerprint": "mining",
                "exported_size": 1,
                "pair_ids": exported["pair_ids"],
                "reviewed_pair_ids_sha256": _pair_ids_hash(exported["pair_ids"]),
                "annotation_sha256": exported["artifact_sha256"],
            }
        ),
        encoding="utf-8",
    )
    row = json.loads(template.read_text(encoding="utf-8"))
    row.update(
        {
            "category": "strict_numeric_length",
            "positive_i_rating": 5,
            "positive_j_rating": 5,
            "cross_i_j_negative_rating": 1,
            "cross_j_i_negative_rating": 1,
            "uncertain": False,
            "rater_id": "human-01",
            "notes": "reviewed from sign videos",
        }
    )
    completed = tmp_path / "completed.jsonl"
    completed.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    decision = {
        "schema_version": "semantic_schema_audit_decision.v1",
        "schema_id": "strict_numeric_v1",
        "audit_id": "audit-2026-01",
        "audit_status": "accepted",
        "proposal_mining_fingerprint": "mining",
        "annotation_path": completed.name,
        "annotation_sha256": hashlib.sha256(completed.read_bytes()).hexdigest(),
        "reviewed_pair_ids_sha256": _pair_ids_hash(exported["pair_ids"]),
        "reviewed_pair_count": 1,
        "rating_scale": "integer_1_to_5_higher_means_caption_matches_sign",
        "decision_author": "principal-investigator",
        "decision_date_utc": "2026-09-08T12:00:00Z",
        "rationale": "All sampled cross-pairs were reviewed.",
        "scope": "strict_numeric_v1_how2sign_train_proposals",
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")
    return template, report, completed, decision_path, decision


def test_human_audit_decision_binds_completed_rows_and_explicit_acceptance(tmp_path):
    template, report, _, decision_path, _ = _write_decision_bundle(tmp_path)
    loaded = load_semantic_audit_decision(
        decision_path,
        export_report_path=report,
        annotation_template_path=template,
        expected_mining_fingerprint="mining",
    )
    assert loaded.schema_audit.audit_status == "accepted"
    assert loaded.schema_audit.artifact_hash == loaded.decision_sha256
    assert loaded.reviewed_pair_count == 1


def test_human_audit_rejects_unfilled_decision_and_changed_immutable_content(tmp_path):
    template, report, completed, decision_path, decision = _write_decision_bundle(tmp_path)
    decision["audit_status"] = None
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(AuditRunError, match="provenance/status"):
        load_semantic_audit_decision(
            decision_path,
            export_report_path=report,
            annotation_template_path=template,
            expected_mining_fingerprint="mining",
        )

    row = json.loads(completed.read_text(encoding="utf-8"))
    row["text_i"] = "changed caption"
    completed.write_text(json.dumps(row) + "\n", encoding="utf-8")
    decision["audit_status"] = "accepted"
    decision["annotation_sha256"] = hashlib.sha256(completed.read_bytes()).hexdigest()
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(AuditRunError, match="changed immutable"):
        load_semantic_audit_decision(
            decision_path,
            export_report_path=report,
            annotation_template_path=template,
            expected_mining_fingerprint="mining",
        )
