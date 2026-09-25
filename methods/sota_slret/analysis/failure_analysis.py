"""Val-split failure analysis for a trained run (never touches official test).

python failure_analysis.py --run <run_dir> --split_dir <split_dir> [--out_md ...] [--out_jsonl ...]
Uses <run>/best_val_sim.npy ([N_video, N_text_group], val dataset order) and <split>/data/test.pkl.
"""
import argparse
import json
import pickle as pkl
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from sanity_eval import evaluate, groups_from_cutoffs  # noqa: E402

TOK = re.compile(r"[a-z0-9äöüß]+")
NUM = re.compile(r"\d+|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                 r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|first|second|third|fourth|fifth|sixth|seventh|eighth|"
                 r"ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|"
                 r"twentieth|thirtieth|minus|degrees?)\b", re.I)


def toks(t):
    return TOK.findall(t.lower())


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def load_val(split_dir):
    val = pkl.load(open(f"{split_dir}/data/test.pkl", "rb"))
    keys = list(val)
    entries = [val[k] if isinstance(val[k], list) else [val[k]] for k in keys]
    texts = [e[0]["text"] for e in entries]
    cut = np.cumsum([len(e) for e in entries]).tolist()
    nframes = [x["num_frames"] for e in entries for x in e]
    return keys, texts, cut, nframes


def bucket(x, edges):
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--sim", default="best_val_sim.npy")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_jsonl", default=None)
    a = ap.parse_args()
    keys, texts, cut, nframes = load_val(a.split_dir)
    S = np.load(f"{a.run}/{a.sim}").astype(np.float64)
    vg, tg = groups_from_cutoffs(cut)
    ev = evaluate(S, vg, tg)
    t2v, v2t = np.array(ev["t2v_ranks"]), np.array(ev["v2t_ranks"])
    n = len(texts)
    T = [toks(t) for t in texts]
    dup = Counter(texts)
    lines = [f"# Failure analysis: `{a.run}` (val, N_text={n}, N_video={len(vg)})", ""]
    lines += ["## Aggregate (sanity evaluator, pessimistic ties)", "",
              "| Dir | R@1 | R@5 | R@10 | MedR | MeanR |", "|---|---:|---:|---:|---:|---:|"]
    for d in ("t2v", "v2t"):
        m = ev[d]
        lines.append(f"| {d.upper()} | {m['R1']:.2f} | {m['R5']:.2f} | {m['R10']:.2f} | {m['MedR']:.1f} | {m['MeanR']:.2f} |")
    lines += ["", "## Rank distribution / top-1 recoverability", ""]
    for d, r in (("T2V", t2v), ("V2T", v2t)):
        hist = {"1": int((r == 1).sum()), "2": int((r == 2).sum()), "3-5": int(((r >= 3) & (r <= 5)).sum()),
                "6-10": int(((r >= 6) & (r <= 10)).sum()), "11-50": int(((r > 10) & (r <= 50)).sum()), ">50": int((r > 50).sum())}
        miss = int((r > 1).sum())
        lines.append(f"- {d}: {hist}; of {miss} top-1 misses, {100*((r>1)&(r<=5)).sum()/max(1,miss):.1f}% have the positive in top-5, "
                     f"{100*((r>1)&(r<=10)).sum()/max(1,miss):.1f}% in top-10.")
    # T2V confusion analysis (texts as queries; gallery groups ordered like texts)
    agg = np.stack([S[vg == g].max(0) for g in range(n)], 0)  # [group(video), text]
    Q = agg.T  # [text query, group]
    rows = []
    for q in range(n):
        order = np.argsort(-Q[q], kind="stable")
        top = [int(g) for g in order[:5]]
        wrong = [g for g in order if g != q][0]
        pos_s, wrong_s = Q[q, q], Q[q, wrong]
        rows.append({"dir": "t2v", "query_idx": q, "query_id": keys[q], "text": texts[q], "rank": int(t2v[q]),
                     "top5_ids": [keys[g] for g in top], "top1_text": texts[top[0]],
                     "hardest_neg_text": texts[wrong], "margin": float(pos_s - wrong_s),
                     "lex_overlap_hardest": jacc(T[q], T[wrong]), "len_tokens": len(T[q]),
                     "n_frames": int(nframes[int(np.where(vg == q)[0][0])]),
                     "same_text_as_neg": texts[q] == texts[wrong], "dup_count": dup[texts[q]],
                     "numbers_differ": set(x.lower() for x in NUM.findall(texts[q])) != set(x.lower() for x in NUM.findall(texts[wrong])) and bool(NUM.findall(texts[q] + texts[wrong]))})
    rows_v = []
    for v in range(len(vg)):
        order = np.argsort(-S[v], kind="stable")
        g = vg[v]
        wrong = [t for t in order if t != g][0]
        rows_v.append({"dir": "v2t", "video_idx": v, "gt": keys[g], "rank": int(v2t[v]), "text": texts[g],
                       "top1_text": texts[order[0]], "hardest_neg_text": texts[wrong],
                       "margin": float(S[v, g] - S[v, wrong]), "lex_overlap_hardest": jacc(T[g], T[wrong]),
                       "same_text_as_neg": texts[g] == texts[wrong]})
    miss = [r for r in rows if r["rank"] > 1]
    hit = [r for r in rows if r["rank"] == 1]
    lines += ["", "## T2V errors: what the top-1 wrong candidate looks like", ""]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    lines.append(f"- lexical Jaccard(query, hardest negative): misses {mean([r['lex_overlap_hardest'] for r in miss]):.3f} vs hits {mean([r['lex_overlap_hardest'] for r in hit]):.3f}")
    lines.append(f"- misses where the hardest negative has the *identical* caption (unresolvable): {sum(r['same_text_as_neg'] for r in miss)} / {len(miss)}")
    lines.append(f"- misses where captions differ in numbers (temperatures, dates…): {sum(r['numbers_differ'] for r in miss)} / {len(miss)} (hits: {sum(r['numbers_differ'] for r in hit)} / {len(hit)})")
    for thr in (0.3, 0.5, 0.7):
        lines.append(f"- misses with Jaccard ≥ {thr}: {sum(r['lex_overlap_hardest'] >= thr for r in miss)} / {len(miss)}")
    lines += ["", "## Buckets (T2V R@1 / V2T R@1 by caption length and video length)", "",
              "| bucket | n | T2V R@1 |", "|---|---:|---:|"]
    for name, vals, edges in (("caption tokens", [r["len_tokens"] for r in rows], [8, 12, 16, 24]),
                              ("video frames", [r["n_frames"] for r in rows], [80, 120, 160, 220])):
        b = [bucket(x, edges) for x in vals]
        for i in range(len(edges) + 1):
            idx = [j for j, bb in enumerate(b) if bb == i]
            if idx:
                lo = "≤%d" % edges[i] if i < len(edges) else ">%d" % edges[-1]
                lines.append(f"| {name} {lo} | {len(idx)} | {100*np.mean(t2v[idx] == 1):.1f} |")
    lines += ["", "## Calibration (T2V margin = s(pos) − s(hardest neg))", ""]
    mg = np.array([r["margin"] for r in rows])
    lines.append(f"- hits: median margin {np.median(mg[t2v == 1]):.3f}; misses: median margin {np.median(mg[t2v > 1]):.3f}")
    lines.append(f"- fraction of misses with |margin| < 0.5 (logit units): {np.mean(np.abs(mg[t2v > 1]) < 0.5):.2f}")
    lines += ["", "## Examples of T2V misses (rank 2–5, highest lexical overlap)", ""]
    for r in sorted([r for r in miss if r["rank"] <= 5], key=lambda r: -r["lex_overlap_hardest"])[:12]:
        lines.append(f"- r={r['rank']} | Q: {r['text']} || top1: {r['top1_text']}")
    if a.out_md:
        open(a.out_md, "w").write("\n".join(lines) + "\n")
    if a.out_jsonl:
        with open(a.out_jsonl, "w") as f:
            for r in [r for r in rows if r["rank"] > 1] + [r for r in rows_v if r["rank"] > 1]:
                f.write(json.dumps(r) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
