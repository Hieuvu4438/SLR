"""Evaluator validation (protocol section 7). Run with the seds env:
    python -m pytest research/sota_slret_agent/tests/test_retrieval_metrics.py -q
Checks upstream SEDS/CiCo metrics (via methods/sota_slret/src/evaluation.official_metrics)
against the independent sanity evaluator on synthetic score matrices with known ranks."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/haipd/SLR/third_party/SEDS")
sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from evaluation import official_metrics, full_metrics  # noqa: E402
from sanity_eval import evaluate, groups_from_cutoffs  # noqa: E402


def single(n):
    return list(range(1, n + 1))  # one video per text group


def test_perfect_retrieval():
    s = np.eye(5) + 0.01 * np.random.RandomState(0).rand(5, 5)
    m = full_metrics(s, single(5))
    for d in ("t2v", "v2t"):
        assert m["official"][d]["R1"] == 100.0
        assert m["sanity"][d]["R1"] == 100.0


def test_reversed_retrieval_known_ranks():
    n = 6
    s = -np.eye(n) + 0.001 * np.arange(n)[None, :]  # positive is always worst
    m = full_metrics(s, single(n))
    assert m["official"]["t2v"]["R1"] == 0.0 and m["official"]["v2t"]["R1"] == 0.0
    assert m["official"]["t2v"]["MR"] == n and m["official"]["v2t"]["MR"] == n
    assert all(r == n for r in m["v2t_ranks"]) and all(r == n for r in m["t2v_ranks"])


def test_orientation_t2v_vs_v2t():
    # 3 videos x 3 texts. Text 0 retrieves video 0 at rank 1, but video 0 prefers text 1.
    s = np.array([[0.9, 1.0, 0.0],
                  [0.1, 0.5, 0.0],
                  [0.0, 0.0, 1.0]])
    m = full_metrics(s, single(3))
    # T2V: columns are queries. text0 -> video0 best (0.9 > 0.1, 0.0): rank1; text1 -> video0 (1.0) beats video1 (0.5): rank2
    assert m["t2v_ranks"] == [1, 2, 1]
    # V2T: rows are queries. video0 -> text1 beats text0: rank2; video1 -> text1 is its... video1 positive text1? no: video1 pos = text1
    assert m["v2t_ranks"] == [2, 1, 1]
    assert m["official"]["t2v"]["R1"] == pytest.approx(200 / 3)
    assert m["official"]["v2t"]["R1"] == pytest.approx(200 / 3)


def test_multi_positive_groups():
    # 4 videos, 2 text groups; videos 0,1 -> group0; videos 2,3 -> group1.
    s = np.array([[0.9, 0.1],
                  [0.2, 0.8],   # video1 prefers wrong text
                  [0.1, 0.7],
                  [0.3, 0.6]])
    cut = [2, 4]
    m = full_metrics(s, cut)
    assert m["v2t_ranks"] == [1, 2, 1, 1]
    assert m["official"]["v2t"]["R1"] == 75.0
    # T2V aggregates group by max: text0 -> g0 max 0.9 vs g1 max 0.3 -> rank1 ; text1 -> g1 0.7 vs g0 0.8 -> rank2
    assert m["t2v_ranks"] == [1, 2]
    assert m["official"]["t2v"]["R1"] == 50.0
    vg, tg = groups_from_cutoffs(cut)
    assert vg.tolist() == [0, 0, 1, 1] and tg.tolist() == [0, 1]


def test_ties_quantified():
    # Exact tie between positive and one negative.
    s = np.array([[1.0, 1.0], [0.0, 1.0]])
    m = full_metrics(s, single(2))
    san = m["sanity"]
    assert san["v2t"]["R1"] == 50.0 and san["v2t_optimistic"]["R1"] == 100.0  # pessimistic vs optimistic
    # Upstream compute_metrics (T2V) counts every tied position => appends extra entries; record behaviour.
    assert "R1" in m["official"]["t2v"]


def test_padding_rows_ignored():
    # Groups of unequal size produce -inf padding in the upstream reshape; must not count as queries.
    s = np.random.RandomState(1).rand(5, 3)
    cut = [1, 3, 5]
    m = official_metrics(s, cut)
    san = evaluate(s, *groups_from_cutoffs(cut))
    assert m["v2t"]["R1"] == pytest.approx(san["v2t"]["R1"])
    assert m["t2v"]["R1"] == pytest.approx(san["t2v"]["R1"])


def test_blockwise_equals_dense():
    rng = np.random.RandomState(2)
    s = rng.rand(40, 40)
    dense = full_metrics(s, single(40))
    blocks = np.concatenate([np.concatenate([s[i:i + 7, j:j + 9] for j in range(0, 40, 9)], 1) for i in range(0, 40, 7)], 0)
    assert np.array_equal(blocks, s)
    assert full_metrics(blocks, single(40))["t2v_ranks"] == dense["t2v_ranks"]


def test_random_agreement_official_vs_sanity():
    rng = np.random.RandomState(3)
    for trial in range(20):
        nt = rng.randint(3, 30)
        sizes = rng.randint(1, 4, size=nt)
        cut = np.cumsum(sizes).tolist()
        s = rng.randn(cut[-1], nt)
        m = full_metrics(s, cut)
        for d in ("t2v", "v2t"):
            for k in ("R1", "R5", "R10"):
                assert m["official"][d][k] == pytest.approx(m["sanity"][d][k]), (trial, d, k)
        # Known upstream quirk: V2T MedR uses torch.median (lower middle value for even n),
        # T2V MedR uses np.median (mean of the two middle values).
        assert m["official"]["t2v"]["MR"] == pytest.approx(m["sanity"]["t2v"]["MedR"])
        r = np.sort(np.array(m["v2t_ranks"]))
        assert m["official"]["v2t"]["MR"] == r[(len(r) - 1) // 2]
