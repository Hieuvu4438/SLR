"""Small, standalone provenance and evaluation utilities (PH train/dev only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/proposal7/phase2'
EVIDENCE = ROOT / 'docs/proposal7/evidence/phase2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rows(split):
    if split not in ('train', 'dev'):
        raise ValueError('Phase 2 permits train/dev only')
    return [json.loads(s) for s in
            (ROOT / f'artifacts/manifests/ph_{split}.jsonl').read_text().splitlines()]


def ranks(scores):
    """Zero-based paired ranks: historical optimistic T2V, torch V2T ties."""
    scores = np.asarray(scores)
    assert scores.ndim == 2 and scores.shape[0] == scores.shape[1]
    assert np.isfinite(scores).all()
    t = (scores > scores.diagonal()[None, :]).sum(0)
    v = torch.argsort(torch.argsort(torch.from_numpy(scores), dim=1, descending=True), dim=1)
    return {'T2V': t, 'V2T': v.diagonal().numpy()}


def persistent():
    historic = [ranks(np.load(ROOT / f'runs/ph_base_b512_s{s}/evaluation/dev/'
                             'scores_video_x_text.npy')) for s in (42, 1337, 2026)]
    return {d: np.stack([r[d] > 0 for r in historic]).all(0) for d in historic[0]}


def metrics(scores, baseline=None):
    r = ranks(scores)
    base = scores if baseline is None else baseline
    br = ranks(base)
    pop = persistent() if len(scores) == 519 else {d: br[d] > 0 for d in br}
    result = {}
    for d, axis in [('T2V', 0), ('V2T', 1)]:
        competitors = base.copy()
        np.fill_diagonal(competitors, -np.inf)
        hard = competitors.argmax(axis=axis)
        idx = np.arange(len(scores))
        wrong = scores[hard, idx] if axis == 0 else scores[idx, hard]
        margin = scores.diagonal() - wrong
        current_wrong = scores.copy()
        np.fill_diagonal(current_wrong, -np.inf)
        best_margin = scores.diagonal() - current_wrong.max(axis=axis)
        persistent_mask = pop[d]
        top10 = persistent_mask & (br[d] < 10)
        result[d] = {
            **{f'R{k}': float(100 * np.mean(r[d] < k)) for k in (1, 5, 10)},
            'MedR': float(np.median(r[d] + 1)), 'MnR': float(np.mean(r[d] + 1)),
            'persistent_n': int(persistent_mask.sum()),
            'persistent_R1': float(100 * np.mean(r[d][persistent_mask] == 0)),
            'persistent_mean_rank_delta': float(np.mean((r[d]-br[d])[persistent_mask])),
            'top10_persistent_n': int(top10.sum()),
            'top10_persistent_R1': float(100 * np.mean(r[d][top10] == 0)),
            'hard_pair_accuracy': float(100 * np.mean(margin[persistent_mask] > 0)),
            'mean_correct_minus_best_incorrect': float(best_margin.mean()),
            'ranks': r[d].tolist(), 'rank_delta': (r[d] - br[d]).tolist(),
            'fixed_hard_candidate': hard.tolist(), 'fixed_hard_margin': margin.tolist(),
            'correct_minus_best_incorrect': best_margin.tolist(),
        }
    result['mean_R1'] = (result['T2V']['R1'] + result['V2T']['R1']) / 2
    result['direction_agreement'] = float(np.mean((r['T2V'] == 0) == (r['V2T'] == 0)))
    # Historical T2V expands every score tie at the positive, unlike the
    # one-rank-per-query diagnostic above. Never silently switch primary metrics.
    expanded = []
    for i in range(len(scores)):
        expanded.extend(np.flatnonzero(np.sort(scores[:, i])[::-1] == scores[i, i]))
    result['official_T2V'] = {
        **{f'R{k}': float(100 * np.mean(np.asarray(expanded) < k)) for k in (1, 5, 10)},
        'MedR': float(np.median(expanded) + 1),
        'MnR': float(np.mean(expanded) + 1), 'rank_entries': len(expanded),
    }
    result['official_mean_R1'] = (result['official_T2V']['R1'] + result['V2T']['R1']) / 2
    return result
