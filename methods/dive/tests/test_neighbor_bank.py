from __future__ import annotations

import numpy as np
import pytest

from dive.mining.audit import export_audit_template
from dive.mining.bank import (
    BankError,
    attach_pair_support,
    finalize_numeric_bank,
    load_bank,
    write_bank,
)
from dive.mining.neighbors import audit_shortlist_coverage, mine_neighbor_proposals
from dive.mining.slots import SchemaAudit
from dive.mining.support import EndpointSupport, PairSupport


def _proposals():
    pooled_video = np.eye(4)
    pooled_text = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.2, 0.8]]
    )
    baseline = np.full((4, 4), -0.2)
    np.fill_diagonal(baseline, 0.8)
    baseline[0, 1] = 0.95
    baseline[1, 0] = 0.90
    return mine_neighbor_proposals(
        pooled_video,
        pooled_text,
        baseline,
        ["s0", "s1", "s2", "s3"],
        np.eye(4, dtype=bool),
        shortlist_per_direction=3,
        rerank_topk=2,
    )


def _audit(status: str = "accepted") -> SchemaAudit:
    return SchemaAudit(
        "strict_numeric_v1",
        status,
        "b" * 64 if status == "accepted" else None,
        "audit_fixture",
    )


def test_two_direction_shortlist_reranks_by_s0_and_deduplicates_pairs():
    proposals = _proposals()
    pairs = [(item.sample_i, item.sample_j) for item in proposals]
    assert len(pairs) == len(set(pairs))
    hard = next(item for item in proposals if (item.sample_i, item.sample_j) == ("s0", "s1"))
    assert hard.hardness == pytest.approx(0.15)
    assert set(hard.directions) == {"t2v", "v2t"}
    assert hard.s0_quartet == ((0.8, 0.95), (0.9, 0.8))


def test_shortlist_coverage_is_measured_against_exact_s0_hardness():
    pooled = np.eye(3)
    baseline = np.array([[0.8, 0.9, -0.2], [0.85, 0.8, -0.2], [-0.2, -0.2, 0.8]])
    report = audit_shortlist_coverage(
        pooled,
        pooled,
        baseline,
        ["s0", "s1", "s2"],
        np.eye(3, dtype=bool),
        ["s0"],
        shortlist_per_direction=1,
        hard_topk=1,
    )
    assert report["query_count"] == 1
    assert report["coverage"] == 1.0


def test_finalize_bank_is_train_only_and_requires_accepted_schema():
    proposals = _proposals()
    captions = {
        "s0": "the value is 12 meters",
        "s1": "the value is 15 meters",
        "s2": "the object is red",
        "s3": "the object is blue",
    }
    splits = {sample: "train" for sample in captions}
    hashes = {sample: f"mapping_{sample}" for sample in captions}
    result = finalize_numeric_bank(
        proposals,
        text_by_sample=captions,
        split_by_sample=splits,
        unit_mapping_hashes=hashes,
        audit=_audit(),
        mining_fingerprint="fingerprint_v1",
    )
    assert [(item.sample_i, item.sample_j) for item in result.records] == [("s0", "s1")]
    assert result.records[0].support_status == "pending_support"
    assert result.records[0].g == 0.0 and result.records[0].g_sem == 1.0
    pending = finalize_numeric_bank(
        proposals,
        text_by_sample=captions,
        split_by_sample=splits,
        unit_mapping_hashes=hashes,
        audit=_audit("pending"),
        mining_fingerprint="fingerprint_v1",
    )
    assert not pending.records and pending.rejected
    splits["s1"] = "dev"
    with pytest.raises(BankError, match="must both be train"):
        finalize_numeric_bank(
            proposals,
            text_by_sample=captions,
            split_by_sample=splits,
            unit_mapping_hashes=hashes,
            audit=_audit(),
            mining_fingerprint="fingerprint_v1",
        )


def test_bank_round_trip_rejects_stale_fingerprint_and_corruption(tmp_path):
    proposals = _proposals()
    captions = {
        "s0": "the value is 12 meters",
        "s1": "the value is 15 meters",
        "s2": "the object is red",
        "s3": "the object is blue",
    }
    build = finalize_numeric_bank(
        proposals,
        text_by_sample=captions,
        split_by_sample={sample: "train" for sample in captions},
        unit_mapping_hashes={sample: f"mapping_{sample}" for sample in captions},
        audit=_audit(),
        mining_fingerprint="fingerprint_v1",
    )
    path = write_bank(build.records, tmp_path, "fingerprint_v1")
    loaded = load_bank(tmp_path, "fingerprint_v1")
    assert loaded[0]["pair_id"] == build.records[0].pair_id
    with pytest.raises(BankError, match="CACHE_HASH_MISMATCH"):
        load_bank(tmp_path, "different")
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(BankError, match="checksum mismatch"):
        load_bank(tmp_path, "fingerprint_v1")


def test_audit_export_is_deterministic_and_never_prefills_human_ratings(tmp_path):
    proposals = _proposals()
    captions = {sample: f"caption {sample}" for sample in ("s0", "s1", "s2", "s3")}
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = export_audit_template(proposals, captions, first_path, sample_size=2, seed=701)
    second = export_audit_template(proposals, captions, second_path, sample_size=2, seed=701)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    rows = [__import__("json").loads(line) for line in first_path.read_text().splitlines()]
    for row in rows:
        assert row["rater_id"] is None
        assert row["positive_i_rating"] is None
        assert row["cross_i_j_negative_rating"] is None
        assert "student" not in row


def test_support_attachment_retains_failed_records_with_zero_weight():
    proposals = _proposals()
    captions = {
        "s0": "the value is 12 meters",
        "s1": "the value is 15 meters",
        "s2": "the object is red",
        "s3": "the object is blue",
    }
    record = finalize_numeric_bank(
        proposals,
        text_by_sample=captions,
        split_by_sample={sample: "train" for sample in captions},
        unit_mapping_hashes={sample: f"mapping_{sample}" for sample in captions},
        audit=_audit(),
        mining_fingerprint="fingerprint_v1",
    ).records[0]
    failed_endpoint = EndpointSupport(None, 0.5, None, 0.0, None, "low_stability")
    failed = attach_pair_support(
        record,
        PairSupport(failed_endpoint, failed_endpoint, 0.0, "endpoint_i:low_stability"),
        reference_hash="reference",
        grid_i="grid_i",
        grid_j="grid_j",
        views_i=("i_left", "i_right"),
        views_j=("j_left", "j_right"),
    )
    assert failed.support_status == "failed"
    assert failed.g == 0 and failed.q_i is None and failed.q_j is None
    assert failed.failure_reason == "endpoint_i:low_stability"


def test_support_attachment_serializes_accepted_q_and_absolute_reliability():
    proposals = _proposals()
    captions = {
        "s0": "the value is 12 meters",
        "s1": "the value is 15 meters",
        "s2": "the object is red",
        "s3": "the object is blue",
    }
    record = finalize_numeric_bank(
        proposals,
        text_by_sample=captions,
        split_by_sample={sample: "train" for sample in captions},
        unit_mapping_hashes={sample: f"mapping_{sample}" for sample in captions},
        audit=_audit(),
        mining_fingerprint="fingerprint_v1",
    ).records[0]
    endpoint = EndpointSupport(np.array([0.7, 0.3]), 0.5, 0.9, 0.7, 0.2, "accepted")
    accepted = attach_pair_support(
        record,
        PairSupport(endpoint, endpoint, 0.45, "accepted"),
        reference_hash="reference",
        grid_i="grid_i",
        grid_j="grid_j",
        views_i=("i_left", "i_right"),
        views_j=("j_left", "j_right"),
    )
    assert accepted.support_status == "accepted" and accepted.failure_reason is None
    assert accepted.g == pytest.approx(0.45)
    assert accepted.q_i == pytest.approx((0.7, 0.3))
