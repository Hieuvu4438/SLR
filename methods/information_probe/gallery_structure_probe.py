"""AS-C43 fixed gallery-only graph filtering diagnostic, not a new method."""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.evaluation.cico_eval import evaluate_score_matrix
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .checkpoint_cross_probe import encode, score
from .common import ART, ROOT, dump, metrics, rows, sha
from .seed_ensemble_probe import OUT, SEEDS, query_counts, mean_r1_from_counts, uniform_mean


def neighbors(descriptors, k=5):
    x = np.asarray(descriptors, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (norm > 0).all() and 0 < k < len(x)
    x = x / norm
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1, kind='stable')[:, :k]


def relabel(graph, permutation):
    p = np.asarray(permutation)
    assert np.array_equal(np.sort(p), np.arange(len(graph)))
    return np.argsort(p)[graph[p]]


def filter_queries(query_x_gallery, graph, beta):
    s = np.asarray(query_x_gallery, dtype=np.float64)
    assert s.ndim == 2 and s.shape[1] == len(graph) and np.isfinite(s).all()
    neighbor_mean = s[:, graph].mean(axis=2)
    return (s + beta * (neighbor_mean - s)).astype(np.float32)


def paired_metrics(pair, base, ids):
    positives = {identity: [identity] for identity in ids}
    result = {}
    for d in ('T2V', 'V2T'):
        s = pair[d]
        m = metrics(s, base[d])
        key = 'official_T2V' if d == 'T2V' else 'V2T'
        official = evaluate_score_matrix(s, video_ids=ids, text_ids=ids,
                                        video_to_text=positives, text_to_video=positives)
        assert all(m[key][f'R{k}'] == official[d][f'R{k}'] for k in (1, 5, 10))
        result[d] = {**m[key], **{name: m[d][name] for name in (
            'ranks', 'persistent_n', 'persistent_R1', 'persistent_mean_rank_delta')}}
    result['mean_R1'] = (result['T2V']['R1'] + result['V2T']['R1']) / 2
    return result


def counts(pair):
    t, v = query_counts(pair['T2V']), query_counts(pair['V2T'])
    return np.concatenate((t[:, :2], v[:, 2:]), axis=1)


def compare(pair, reference, labels):
    labels = np.asarray(labels)
    groups = np.unique(labels)
    a, b = counts(pair), counts(reference)
    ac = np.stack([a[labels == g].sum(0) for g in groups])
    bc = np.stack([b[labels == g].sum(0) for g in groups])
    take = np.random.default_rng(20260915).integers(len(groups), size=(10000, len(groups)))
    delta = mean_r1_from_counts(ac[take].sum(1)) - mean_r1_from_counts(bc[take].sum(1))
    return {'mean_pp': float(mean_r1_from_counts(a.sum(0)) - mean_r1_from_counts(b.sum(0))),
            'CI95': np.quantile(delta, [.025, .975]).tolist(),
            'CI_bonferroni16': np.quantile(delta, [.025/16, 1-.025/16]).tolist(),
            'draws': 10000, 'clusters': len(groups), 'seed': 20260915,
            'scope': 'fixed full519 gallery; inferred source prefixes; reused exploratory DEV; no selection/training uncertainty'}


def gate(m, base_m, ci, shuffle_ci):
    result = {'gain': ci['mean_pp'] >= .5, 'CI': ci['CI95'][0] > 0,
              'R1': all(m[d]['R1'] - base_m[d]['R1'] >= -.25 for d in ('T2V', 'V2T')),
              'R5_R10': all(m[d][f'R{k}'] - base_m[d][f'R{k}'] >= -.5 for d in ('T2V', 'V2T') for k in (5, 10)),
              'persistent': all(m[d]['persistent_mean_rank_delta'] < 0 for d in ('T2V', 'V2T')),
              'vs_shuffle': shuffle_ci['CI95'][0] > 0}
    result['passes'] = all(result.values())
    return result


@torch.inference_mode()
def main():
    target = OUT/'AS-C43-GALLERY_run.json'
    if target.exists():
        raise FileExistsError(target)
    artifact = ART/'AS-C43'
    artifact.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    started = time.time()
    report = {'experiment_id': 'AS-C43-GALLERY', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C43_protocol.md'),
              'updates': 0, 'trainable_parameters': 0, 'method_go': False, 'test_loaded': False,
              'checkpoints': {}, 'conditions': {}, 'comparisons': {}, 'gates': {},
              'environment': {'torch': torch.__version__, 'numpy': np.__version__,
                              'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda}}
    dump(target, report)

    def save(name, array):
        path = artifact/f'{name}.npy'
        if path.exists():
            raise FileExistsError(path)
        np.save(path, array)
        return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}

    try:
        parent_path = OUT/'AS-C32-ENSEMBLE_run.json'
        parent = json.loads(parent_path.read_text())
        assert parent['status'] == 'completed'
        report['parent_sha256'] = sha(parent_path)
        manifest = ROOT/'artifacts/manifests/ph_dev.jsonl'
        assert sha(manifest) == parent['manifest_sha256']
        report['manifest_sha256'] = sha(manifest)
        rec = rows('dev')
        ids = [x['pair_id'] for x in rec]
        labels = [x['video_id'].rsplit('-', 1)[0] for x in rec]
        permutation = np.random.default_rng(20260915).permutation(len(ids))
        report['permutation'] = permutation.tolist()
        all_pairs, baselines = {}, {}
        for seed in SEEDS:
            if time.time()-started > 550:
                raise TimeoutError('AS-C43 internal550s')
            run = ROOT/f'runs/ph_base_b512_s{seed}'
            cp = run/'checkpoints/best_dev.pt'
            assert sha(cp) == parent['inputs'][str(seed)]['checkpoint_sha256']
            config_path = run/'resolved_config.yaml'
            config = yaml.safe_load(config_path.read_text())
            cico_root = ROOT/config['upstream']['cico_root']
            sys.path.insert(0, str(cico_root))
            saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
            state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
            core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
            core.eval().requires_grad_(False)
            loader = DataLoader(CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split='dev'),
                                batch_size=128, num_workers=4, shuffle=False,
                                collate_fn=CiCoCollator(load_cico_tokenizer(config), 32, augment=False))
            v, t, actual_ids = encode(core, loader)
            assert actual_ids == ids
            original_path = run/'evaluation/dev/scores_video_x_text.npy'
            assert sha(original_path) == parent['inputs'][str(seed)]['score_sha256']
            original = np.load(original_path)
            assert np.array_equal(score(CiCoBridge(core), v, t), original)
            base = baselines[str(seed)] = {d: original for d in ('T2V', 'V2T')}
            graphs, assets = {}, {}
            for d, encoding, valid_value in [('T2V', v, 0), ('V2T', t, 1)]:
                valid = encoding.mask == valid_value
                descriptors = (encoding.tokens.float().masked_fill(~valid[..., None], 0).sum(1)
                               / valid.sum(1)[:, None]).cpu().numpy()
                graph = neighbors(descriptors)
                shuffled = relabel(graph, permutation)
                graphs[d] = {'real': graph, 'shuffle': shuffled}
                assert not (graph == np.arange(len(ids))[:, None]).any()
                assert not (shuffled == np.arange(len(ids))[:, None]).any()
                assert np.array_equal(np.sort(np.bincount(graph.ravel(), minlength=len(ids))),
                                      np.sort(np.bincount(shuffled.ravel(), minlength=len(ids))))
                assets[d] = {'descriptors': save(f's{seed}-{d}-descriptors', descriptors),
                             'real_graph': save(f's{seed}-{d}-real-graph', graph),
                             'shuffle_graph': save(f's{seed}-{d}-shuffle-graph', shuffled)}
            report['checkpoints'][str(seed)] = {'checkpoint_sha256': sha(cp), 'config_sha256': sha(config_path),
                                               'baseline_scores_exact': True, 'assets': assets}
            for operator, beta in [('smooth', .5), ('sharp', -.5)]:
                for kind in ('real', 'shuffle'):
                    name = f's{seed}-{operator}-{kind}'
                    pair, files = {}, {}
                    for d in ('T2V', 'V2T'):
                        query_scores = original.T if d == 'T2V' else original
                        graph = graphs[d][kind]
                        filtered = filter_queries(query_scores, graph, beta)
                        singles = np.concatenate([filter_queries(x[None], graph, beta) for x in query_scores])
                        assert np.array_equal(filtered, singles)
                        s = filtered.T.copy() if d == 'T2V' else filtered
                        pair[d] = s
                        files[d] = save(f'{name}-{d}-scores', s)
                    all_pairs[name] = pair
                    report['conditions'][name] = {'metrics': paired_metrics(pair, base, ids),
                                                'scores': files, 'single_query_exact': True}
            dump(target, report)
            print(json.dumps({'seed': seed, 'baseline_exact': True,
                              'mean_R1': {k: x['metrics']['mean_R1'] for k, x in report['conditions'].items() if k.startswith(f's{seed}-')}}), flush=True)
            del core, saved, state, v, t, loader
            torch.cuda.empty_cache()
        ensemble_path = ART/'AS-C32-uniform3_scores.npy'
        assert sha(ensemble_path) == parent['ensemble_score_sha256']
        ensemble = np.load(ensemble_path)
        assert np.array_equal(uniform_mean([baselines[str(s)]['T2V'] for s in SEEDS]), ensemble)
        baselines['ensemble'] = {d: ensemble for d in ('T2V', 'V2T')}
        for operator in ('smooth', 'sharp'):
            for kind in ('real', 'shuffle'):
                name = f'ensemble-{operator}-{kind}'
                pair = {d: uniform_mean([all_pairs[f's{seed}-{operator}-{kind}'][d] for seed in SEEDS]) for d in ('T2V', 'V2T')}
                all_pairs[name] = pair
                report['conditions'][name] = {'metrics': paired_metrics(pair, baselines['ensemble'], ids),
                    'scores': {d: save(f'{name}-{d}-scores', s) for d, s in pair.items()}}
        for source in [str(s) for s in SEEDS] + ['ensemble']:
            prefix = source if source == 'ensemble' else f's{source}'
            base = baselines[source]
            bm = paired_metrics(base, base, ids)
            for operator in ('smooth', 'sharp'):
                name, shuffled = f'{prefix}-{operator}-real', f'{prefix}-{operator}-shuffle'
                ci = compare(all_pairs[name], base, labels)
                sci = compare(all_pairs[name], all_pairs[shuffled], labels)
                report['comparisons'][name] = {'vs_baseline': ci, 'vs_shuffle': sci}
                report['gates'][name] = gate(report['conditions'][name]['metrics'], bm, ci, sci)
        report['diagnostic_leads'] = {operator: (
            sum(report['gates'][f's{s}-{operator}-real']['passes'] for s in SEEDS) >= 2
            and report['gates'][f'ensemble-{operator}-real']['passes']) for operator in ('smooth', 'sharp')}
        report.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(target, report)
        print(json.dumps({'status': report['status'], 'comparisons': report['comparisons'],
                          'gates': report['gates'], 'diagnostic_leads': report['diagnostic_leads'],
                          'wall_seconds': report['wall_seconds']}, indent=2), flush=True)
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(target, report)
        raise


if __name__ == '__main__':
    main()
