"""Independent deterministic retrieval evaluator (sanity path, not the headline path).

Inputs are a raw score matrix S[video, text] plus a group id per video and per text.
A text is a correct match for a video iff their group ids are equal (multi-positive safe).

T2V (per text query): rank = 1 + #distinct groups... no -- rank over the *gallery items*:
  gallery = videos aggregated per group by max score (matches upstream CiCo/SEDS T2V).
V2T (per video query): gallery = unique texts (one per group, matching upstream), rank of
  the positive text.
Ties are resolved pessimistically (rank counts every gallery item with score >= positive,
excluding the positive itself), and an optimistic rank is also returned so the tie effect
can be quantified.
"""
import numpy as np


def _ranks(scores, pos_idx):
    """scores: [Q, G]; pos_idx: [Q]. Returns (pessimistic, optimistic) 1-based ranks."""
    q = np.arange(scores.shape[0])
    pos = scores[q, pos_idx][:, None]
    greater = (scores > pos).sum(1)
    ties = (scores == pos).sum(1) - 1
    return greater + ties + 1, greater + 1


def retrieval_metrics(ranks):
    ranks = np.asarray(ranks)
    return {"R1": 100.0 * np.mean(ranks <= 1), "R5": 100.0 * np.mean(ranks <= 5),
            "R10": 100.0 * np.mean(ranks <= 10), "MedR": float(np.median(ranks)),
            "MeanR": float(np.mean(ranks)), "n": int(len(ranks))}


def evaluate(sim_vt, video_group, text_group):
    """sim_vt: [Nv, Nt]; text_group must be unique per text (one text per group)."""
    sim_vt = np.asarray(sim_vt, dtype=np.float64)
    video_group = np.asarray(video_group)
    text_group = np.asarray(text_group)
    assert sim_vt.shape == (len(video_group), len(text_group))
    assert len(set(text_group.tolist())) == len(text_group), "one text per group expected"
    assert np.isfinite(sim_vt).all()
    t_index = {g: i for i, g in enumerate(text_group.tolist())}
    # V2T: every video queries the unique-text gallery.
    v_pos = np.array([t_index[g] for g in video_group.tolist()])
    v2t_pes, v2t_opt = _ranks(sim_vt, v_pos)
    # T2V: every text queries the group-max-aggregated video gallery (ordered as text_group).
    agg = np.full((len(text_group), len(text_group)), -np.inf)
    for g, j in t_index.items():
        rows = np.where(video_group == g)[0]
        assert len(rows) > 0, f"group {g} has no video"
        agg[j] = sim_vt[rows].max(0)
    t2v_pes, t2v_opt = _ranks(agg.T, np.arange(len(text_group)))
    return {
        "t2v": retrieval_metrics(t2v_pes), "v2t": retrieval_metrics(v2t_pes),
        "t2v_optimistic": retrieval_metrics(t2v_opt), "v2t_optimistic": retrieval_metrics(v2t_opt),
        "t2v_ranks": t2v_pes, "v2t_ranks": v2t_pes,
    }


def groups_from_cutoffs(cut_off_points):
    """Upstream eval datasets store cumulative video counts per text group (cut_off_points)."""
    groups, start = [], 0
    for g, end in enumerate(cut_off_points):
        groups += [g] * (end - start)
        start = end
    return np.array(groups), np.arange(len(cut_off_points))
