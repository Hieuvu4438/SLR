from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dive.adapters import seds_prelogit_fusion_scores
from dive.mining.neighbors import NeighborError, audit_shortlist_coverage, mine_neighbor_proposals
from dive.mining.pair_scorer import PersistentSedsPairScorer
from dive.mining.proposals import load_proposals, write_proposals
from dive.mining.scalable import (
    audit_sparse_shortlist_coverage,
    pooled_shortlists_blockwise,
    rerank_sparse_shortlists,
)


def _fixture():
    video = np.eye(4, dtype=np.float32)
    text = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.2, 0.8],
        ],
        dtype=np.float32,
    )
    baseline = np.full((4, 4), -0.2, dtype=np.float32)
    np.fill_diagonal(baseline, 0.8)
    baseline[0, 1] = 0.95
    baseline[1, 0] = 0.90
    ids = ("s0", "s1", "s2", "s3")
    forbidden = tuple({index} for index in range(4))
    return video, text, baseline, ids, forbidden


def test_blockwise_shortlist_and_sparse_rerank_match_dense_oracle():
    video, text, baseline, ids, forbidden = _fixture()
    shortlists = pooled_shortlists_blockwise(
        torch.from_numpy(video),
        torch.from_numpy(text),
        ids,
        forbidden,
        limit=3,
        query_chunk_size=2,
        device="cpu",
    )

    def score(video_indices, text_indices):
        return baseline[video_indices, text_indices]

    sparse = rerank_sparse_shortlists(
        shortlists,
        ids,
        forbidden,
        score,
        rerank_topk=2,
    )
    dense = mine_neighbor_proposals(
        video,
        text,
        baseline,
        ids,
        np.eye(4, dtype=bool),
        shortlist_per_direction=3,
        rerank_topk=2,
    )
    assert sparse.proposals == dense
    np.testing.assert_allclose(sparse.self_scores, np.diag(baseline))
    assert sparse.shortlisted_pair_count == 6

    sparse_audit = audit_sparse_shortlist_coverage(
        shortlists,
        ids,
        forbidden,
        ["s0", "s2"],
        score,
        sparse.self_scores,
        hard_topk=2,
    )
    dense_audit = audit_shortlist_coverage(
        video,
        text,
        baseline,
        ids,
        np.eye(4, dtype=bool),
        ["s0", "s2"],
        shortlist_per_direction=3,
        hard_topk=2,
    )
    assert sparse_audit == dense_audit


def test_blockwise_shortlist_breaks_boundary_ties_by_sample_id():
    pooled = torch.ones(4, 2)
    ids = ("z", "a", "m", "b")
    shortlists = pooled_shortlists_blockwise(
        pooled,
        pooled,
        ids,
        tuple({index} for index in range(4)),
        limit=2,
        query_chunk_size=3,
        device="cpu",
    )
    assert shortlists.video_to_text[0].tolist() == [1, 3]
    assert shortlists.text_to_video[0].tolist() == [1, 3]


def test_sparse_rerank_requires_both_cross_pairs_to_be_eligible():
    video, text, baseline, ids, forbidden = _fixture()
    asymmetric = list(forbidden)
    asymmetric[1] = {0, 1}
    shortlists = pooled_shortlists_blockwise(
        torch.from_numpy(video),
        torch.from_numpy(text),
        ids,
        asymmetric,
        limit=2,
        query_chunk_size=2,
        device="cpu",
    )
    result = rerank_sparse_shortlists(
        shortlists,
        ids,
        asymmetric,
        lambda rows, columns: baseline[rows, columns],
        rerank_topk=2,
    )
    assert all(
        {proposal.sample_i, proposal.sample_j} != {"s0", "s1"} for proposal in result.proposals
    )


def test_neighbor_proposals_round_trip_with_fingerprint_and_checksum(tmp_path):
    video, text, baseline, ids, forbidden = _fixture()
    shortlists = pooled_shortlists_blockwise(
        torch.from_numpy(video),
        torch.from_numpy(text),
        ids,
        forbidden,
        limit=3,
        query_chunk_size=2,
        device="cpu",
    )
    proposals = rerank_sparse_shortlists(
        shortlists,
        ids,
        forbidden,
        lambda rows, columns: baseline[rows, columns],
        rerank_topk=2,
    ).proposals
    write_proposals(proposals, tmp_path, mining_fingerprint="fingerprint")
    assert load_proposals(
        tmp_path,
        expected_fingerprint="fingerprint",
        expected_sample_ids=ids,
    ) == tuple(sorted(proposals, key=lambda item: item.pair_id))
    with pytest.raises(NeighborError, match="CACHE_HASH_MISMATCH"):
        load_proposals(tmp_path, expected_fingerprint="stale")
    with (tmp_path / "proposals.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(NeighborError, match="checksum mismatch"):
        load_proposals(tmp_path, expected_fingerprint="fingerprint")


def test_persistent_pair_scorer_matches_cartesian_and_resumes_at_chunk_boundary(tmp_path):
    generator = torch.Generator().manual_seed(229)
    video = torch.randn(4, 3, 5, generator=generator)
    text = torch.randn(4, 2, 5, generator=generator)
    video_mask = torch.tensor(
        [[True, True, True], [True, True, False], [True, False, False], [True, True, False]]
    )
    text_mask = torch.tensor([[True, True], [True, False], [True, True], [True, False]])
    scorer = PersistentSedsPairScorer(
        video,
        text,
        video_mask,
        text_mask,
        dual_mix=0.3,
        temperature=0.07,
        device="cpu",
        pair_chunk_size=2,
        cache_dir=tmp_path,
        fingerprint="features",
    )
    video_indices = np.asarray([0, 0, 3, 2])
    text_indices = np.asarray([1, 3, 0, 2])
    expected = seds_prelogit_fusion_scores(
        video,
        text,
        video_mask,
        text_mask,
        dual_mix=0.3,
    )[0][video_indices, text_indices]
    np.testing.assert_allclose(scorer(video_indices, text_indices), expected.numpy(), atol=1e-6)

    state_path = next(tmp_path.glob("scores-*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completed"] = 2
    state["scores_sha256"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    np.testing.assert_allclose(scorer(video_indices, text_indices), expected.numpy(), atol=1e-6)

    with next(tmp_path.glob("scores-*.npy")).open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(NeighborError, match="checksum mismatch"):
        scorer(video_indices, text_indices)
