"""Post-hoc (K, w) sweep of the cmr reranker on VAL using saved artifacts (no GPU).
python rerank_sweep.py --run <cmr run dir> [--cut_from <split_dir>]
w=0 row = the same run's bi-encoder alone (the key same-checkpoint control)."""
import argparse
import json
import pickle as pkl
import sys

import numpy as np

sys.path[:0] = ["/home/haipd/SLR/third_party/SEDS", "/home/haipd/SLR/methods/sota_slret/src"]
from evaluation import full_metrics  # noqa: E402
from method_impls import rerank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split_dir", required=True)
    a = ap.parse_args()
    val = pkl.load(open(f"{a.split_dir}/data/test.pkl", "rb"))
    cut = np.cumsum([len(v) if isinstance(v, list) else 1 for v in val.values()]).tolist()
    S = np.load(f"{a.run}/best_val_sim_stream_fusion.npy").astype(np.float64)
    art = dict(np.load(f"{a.run}/best_val_rerank.npz"))
    base = full_metrics(S, cut)
    rows = []
    for k in (0, 4, 8, 16, 32):
        for w in (0.25, 0.5, 1.0, 2.0, 4.0, 1e3):
            if k > art["top_v"].shape[1]:
                continue
            St, Sv = rerank(S, art, k, w)
            m = full_metrics(St, cut, Sv)
            rows.append({"k": k, "w": w, "t2v": m["official"]["t2v"]["R1"], "v2t": m["official"]["v2t"]["R1"],
                         "primary": m["primary"],
                         "t2v_fixed": int(np.sum((np.array(m["t2v_ranks"]) == 1) & (np.array(base["t2v_ranks"]) > 1))),
                         "t2v_broken": int(np.sum((np.array(m["t2v_ranks"]) > 1) & (np.array(base["t2v_ranks"]) == 1)))})
            if k == 0:
                break
    print(f"bi-encoder only (w=0): T2V {base['official']['t2v']['R1']:.2f} V2T {base['official']['v2t']['R1']:.2f} primary {base['primary']:.2f}")
    for r in sorted(rows, key=lambda r: -r["primary"])[:12]:
        print(json.dumps(r))
    # itm-only ordering inside top-k (w huge) tells whether the head alone ranks well
    json.dump(rows, open(f"{a.run}/rerank_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
