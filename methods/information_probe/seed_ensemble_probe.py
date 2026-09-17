"""AS-C32: fixed independent-query ensemble control, no fitting or novelty claim."""
import json
import os
import time
import traceback

import numpy as np

from slr_common.evaluation.cico_eval import evaluate_score_matrix
from .common import ART, ROOT, dump, metrics, ranks, rows, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'
SEEDS = (42, 1337, 2026)


def uniform_mean(scores):
    assert len(scores) > 0 and all(x.shape == scores[0].shape for x in scores)
    return np.mean(np.stack(scores).astype(np.float64), axis=0).astype(np.float32)


def query_counts(scores):
    """Official numerator/denominator contributions, preserving expanded T ties."""
    diag = scores.diagonal()
    t_den = (scores == diag[None, :]).sum(0)
    t_num = (scores > diag[None, :]).sum(0) == 0
    v_num = ranks(scores)['V2T'] == 0
    return np.stack((t_num, t_den, v_num, np.ones(len(scores))), 1).astype(float)


def mean_r1_from_counts(c):
    return 50*(c[..., 0]/c[..., 1] + c[..., 2]/c[..., 3])


def bootstrap_delta(candidate, baseline, labels, draws=10000):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    a, b = query_counts(candidate), query_counts(baseline)
    ac = np.stack([a[labels == g].sum(0) for g in unique])
    bc = np.stack([b[labels == g].sum(0) for g in unique])
    rng = np.random.default_rng(20260915)
    take = rng.integers(len(unique), size=(draws, len(unique)))
    dist = mean_r1_from_counts(ac[take].sum(1))-mean_r1_from_counts(bc[take].sum(1))
    return {'mean_pp': float(mean_r1_from_counts(a.sum(0))-mean_r1_from_counts(b.sum(0))),
            'lower': float(np.quantile(dist, .025)), 'upper': float(np.quantile(dist, .975)),
            'draws': draws, 'clusters': len(unique), 'bootstrap_seed': 20260915,
            'scope': 'one fixed ensemble versus fixed best single model, conditional on full519 gallery',
            'unit': 'inferred filename prefix, not verified recording identity',
            'tie_expansion_preserved': True}


def main():
    path = OUT/'AS-C32-ENSEMBLE_run.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.time()
    result = {'experiment_id': 'AS-C32-ENSEMBLE', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C32_protocol.md'),
              'updates': 0, 'method_go': False, 'test_loaded': False, 'inputs': {}, 'variants': {}}
    dump(path, result)
    try:
        records = rows('dev')
        ids = [r['pair_id'] for r in records]
        labels = [r['video_id'].rsplit('-', 1)[0] for r in records]
        positives = {i: [i] for i in ids}
        result['manifest_sha256'] = sha(ROOT/'artifacts/manifests/ph_dev.jsonl')
        scores = {}
        for seed in SEEDS:
            root = ROOT/f'runs/ph_base_b512_s{seed}'
            score_path = root/'evaluation/dev/scores_video_x_text.npy'
            metadata_path = root/'evaluation/dev/metrics.json'
            meta = json.loads(metadata_path.read_text())
            assert meta['score_orientation'] == 'video_x_text' and meta['split'] == 'dev'
            for d in ('T2V', 'V2T'):
                assert [x['query_id'] for x in meta['per_query'][d]] == ids
            cp = root/'checkpoints/best_dev.pt'
            assert sha(cp) == meta['checkpoint']['sha256']
            s = np.load(score_path)
            assert s.shape == (519, 519) and np.isfinite(s).all()
            scores[seed] = s
            m = metrics(s, scores[42])
            for d, key in (('T2V', 'official_T2V'), ('V2T', 'V2T')):
                assert all(m[key][f'R{k}'] == meta[d][f'R{k}'] for k in (1, 5, 10))
            result['inputs'][str(seed)] = {'score_sha256': sha(score_path), 'metadata_sha256': sha(metadata_path),
                                         'checkpoint_sha256': sha(cp), 'id_hashes': meta['id_hashes']}
            result['variants'][f'single_{seed}'] = m
        assert all(result['inputs'][str(s)]['id_hashes'] == result['inputs']['42']['id_hashes'] for s in SEEDS)
        baseline = scores[42]
        assert np.array_equal(baseline, np.load(ART/'baseline_dev_scores.npy'))
        best = max(SEEDS, key=lambda s: result['variants'][f'single_{s}']['official_mean_R1'])
        assert best == 42
        for seed in SEEDS:
            repeat = uniform_mean([scores[seed]]*3)
            assert np.array_equal(repeat, scores[seed])
            result['variants'][f'repeat3_{seed}'] = metrics(repeat, baseline)
        ensemble = uniform_mean([scores[s] for s in SEEDS])
        exact_mean = np.stack([scores[s].astype(np.float64) for s in SEEDS]).mean(0)
        result['float32_rounding_rank_changes'] = {d: int((ranks(ensemble)[d] != ranks(exact_mean)[d]).sum()) for d in ('T2V', 'V2T')}
        assert not any(result['float32_rounding_rank_changes'].values())
        m = metrics(ensemble, baseline)
        official = evaluate_score_matrix(ensemble, video_ids=ids, text_ids=ids,
                                         video_to_text=positives, text_to_video=positives)
        for d, key in (('T2V', 'official_T2V'), ('V2T', 'V2T')):
            assert all(m[key][f'R{k}'] == official[d][f'R{k}'] for k in (1, 5, 10))
        result['variants']['uniform3'] = m
        target = ART/'AS-C32-uniform3_scores.npy'
        if target.exists():
            raise FileExistsError(target)
        np.save(target, ensemble)
        result['ensemble_score_sha256'] = sha(target)
        result['complementarity'] = {}
        for d, axis in (('T2V', 0), ('V2T', 1)):
            all_ranks = np.stack([ranks(scores[s])[d] for s in SEEDS])
            persistent = (all_ranks > 0).all(0)
            recovered = persistent & (ranks(ensemble)[d] == 0)
            strict = np.stack([scores[s] > (scores[s].diagonal()[None, :] if axis == 0 else scores[s].diagonal()[:, None]) for s in SEEDS])
            common_strict = strict.all(0).any(axis=axis)
            assert not (recovered & common_strict).any()
            result['complementarity'][d] = {
                'persistent_n': int(persistent.sum()), 'recovered_persistent_n': int(recovered.sum()),
                'recovered_persistent_indexes': np.flatnonzero(recovered).tolist(),
                'persistent_common_strict_confuser_n': int((persistent & common_strict).sum()),
                'persistent_without_common_strict_confuser_n': int((persistent & ~common_strict).sum()),
                'correct_in_all_single_models_but_lost_n': int(((all_ranks == 0).all(0) & (ranks(ensemble)[d] > 0)).sum())}
        bm = result['variants']['single_42']
        ci = bootstrap_delta(ensemble, baseline, labels)
        result['bootstrap_vs_best_single'] = ci
        assert abs(ci['mean_pp']-(m['official_mean_R1']-bm['official_mean_R1'])) < 1e-12
        gate = {'mean_gain_at_least_half_pp': ci['mean_pp'] >= .5, 'bootstrap_lower_positive': ci['lower'] > 0,
                'R1_nonregression': all(m[d]['R1']-bm[d]['R1'] >= -.25 for d in ('official_T2V', 'V2T')),
                'R5_R10_nonregression': all(m[d][f'R{k}']-bm[d][f'R{k}'] >= -.5 for d in ('official_T2V', 'V2T') for k in (5, 10)),
                'persistent_ranks_improve_both': all(m[d]['persistent_mean_rank_delta'] < 0 for d in ('T2V', 'V2T'))}
        gate['diagnostic_lead'] = all(gate.values())
        result.update(gate=gate, status='completed', wall_seconds=time.time()-started,
                      inference_models=3, model_encoder_scorer_compute_multiplier=3,
                      independent_ensemble_replicates=1, ordinary_method_go=False)
        dump(path, result)
        print(json.dumps({'status': result['status'], 'mean_R1': {k: x['official_mean_R1'] for k, x in result['variants'].items()},
                          'bootstrap': ci, 'gate': gate, 'complementarity': result['complementarity'], 'wall_seconds': result['wall_seconds']}, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
