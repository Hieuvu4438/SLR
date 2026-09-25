"""Paired bootstrap over val sentence groups for R@1 differences between two score matrices."""
import sys, numpy as np
sys.path[:0] = ["/home/haipd/SLR/third_party/SEDS", "/home/haipd/SLR/methods/sota_slret/src"]
from sanity_eval import evaluate, groups_from_cutoffs
def ranks(S, cut):
    e = evaluate(S, *groups_from_cutoffs(cut)); return np.array(e["t2v_ranks"]), np.array(e["v2t_ranks"]), groups_from_cutoffs(cut)[0]
def compare(SA, SB, cut, B=10000, seed=0):
    ta, va, vg = ranks(SA, cut); tb, vb, _ = ranks(SB, cut)
    rng = np.random.default_rng(seed); G = len(cut)
    d_t = (tb == 1).astype(float) - (ta == 1); d_v = (vb == 1).astype(float) - (va == 1)
    # V2T queries are videos; bootstrap unit = sentence group (videos of a group resampled together)
    gv = [np.where(vg == g)[0] for g in range(G)]
    bt, bv = [], []
    for _ in range(B):
        idx = rng.integers(0, G, G)
        bt.append(100 * d_t[idx].mean()); bv.append(100 * np.concatenate([d_v[gv[g]] for g in idx]).mean())
    bt, bv = np.array(bt), np.array(bv); bp = (bt + bv) / 2
    out = {"t2v_delta": 100 * d_t.mean(), "t2v_ci95": np.percentile(bt, [2.5, 97.5]).round(2).tolist(),
           "v2t_delta": 100 * d_v.mean(), "v2t_ci95": np.percentile(bv, [2.5, 97.5]).round(2).tolist(),
           "primary_delta": 100 * (d_t.mean() + d_v.mean()) / 2, "primary_ci95": np.percentile(bp, [2.5, 97.5]).round(2).tolist(),
           "p_primary_le_0": float((bp <= 0).mean()),
           "t2v_improved/worsened": (int((d_t > 0).sum()), int((d_t < 0).sum())), "v2t_improved/worsened": (int((d_v > 0).sum()), int((d_v < 0).sum()))}
    return out
if __name__ == "__main__":
    import pickle as pkl, json
    a, b, split = sys.argv[1:4]
    val = pkl.load(open(f"{split}/data/test.pkl", "rb"))
    cut = np.cumsum([len(v) if isinstance(v, list) else 1 for v in val.values()]).tolist()
    ld = lambda p: (np.load(p)["sim_fusion"] if p.endswith(".npz") else np.load(p)).astype(np.float64)
    print(json.dumps(compare(ld(a), ld(b), cut), indent=1))
