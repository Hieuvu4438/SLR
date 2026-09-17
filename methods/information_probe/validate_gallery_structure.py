"""Independent scalar-neighbor reconstruction and official evaluation of AS-C43."""
import json
import time

import numpy as np

from slr_common.evaluation.cico_eval import evaluate_score_matrix
from .common import ART, ROOT, dump, ranks, rows, sha
from .seed_ensemble_probe import OUT, SEEDS


def scalar_candidates(query_scores, graph, beta):
    """Independent candidate/neighbor loops; no vectorized gather or probe imports."""
    original = query_scores.astype(np.float64)
    result = np.empty_like(query_scores)
    for candidate, indexes in enumerate(graph):
        total = np.zeros(len(query_scores), dtype=np.float64)
        for neighbor in indexes:
            total += original[:, int(neighbor)]
        result[:, candidate] = original[:, candidate] + beta * (
            total / len(indexes) - original[:, candidate])
    return result


def main():
    output = OUT/'AS-C43-VALIDATION_run.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    parent_path = OUT/'AS-C43-GALLERY_run.json'
    parent = json.loads(parent_path.read_text())
    assert parent['status'] == 'completed'
    assert sha(ROOT/'methods/information_probe/gallery_structure_probe.py') == parent['code_sha256']
    assert sha(OUT/'AS-C43_protocol.md') == parent['protocol_sha256']
    assert sha(ROOT/'artifacts/manifests/ph_dev.jsonl') == parent['manifest_sha256']
    report = {'status': 'validating', 'parent_sha256': sha(parent_path), 'code_sha256': sha(__file__),
              'scope': 'independent saved-descriptor neighbor certificate, scalar score reconstruction and official metrics; no encoder rerun',
              'graphs': {}, 'conditions': {}}
    rec = rows('dev')
    ids = [r['pair_id'] for r in rec]
    positives = {i: [i] for i in ids}
    p = np.asarray(parent['permutation'])
    assert np.array_equal(np.sort(p), np.arange(len(ids)))
    all_scores = {}

    def load(asset):
        path = ROOT/asset['path']
        assert path.resolve().parent == ART/'AS-C43'
        assert sha(path) == asset['sha256']
        return np.load(path)

    for seed in SEEDS:
        baseline = np.load(ROOT/f'runs/ph_base_b512_s{seed}/evaluation/dev/scores_video_x_text.npy')
        for d in ('T2V', 'V2T'):
            assets = parent['checkpoints'][str(seed)]['assets'][d]
            descriptors = load(assets['descriptors']).astype(np.float64)
            x = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
            graph, shuffled = load(assets['real_graph']), load(assets['shuffle_graph'])
            assert graph.shape == shuffled.shape == (519, 5)
            minimum_gap = float('inf')
            for i, selected in enumerate(graph):
                assert len(set(selected)) == 5 and i not in selected
                cosine = (x * x[i]).sum(axis=1)
                excluded = np.ones(len(ids), dtype=bool)
                excluded[selected] = False
                excluded[i] = False
                gap = float(cosine[selected].min() - cosine[excluded].max())
                minimum_gap = min(minimum_gap, gap)
                assert gap >= -1e-12
                assert np.all(np.diff(cosine[selected]) <= 1e-12)
            # Full adjacency conjugation, independently of inverse-index helper.
            adj = np.zeros((519, 519), dtype=np.int8)
            shuffled_adj = np.zeros_like(adj)
            for i in range(519):
                adj[i, graph[i]] = 1
                shuffled_adj[i, shuffled[i]] = 1
            assert np.array_equal(shuffled_adj, adj[np.ix_(p, p)])
            report['graphs'][f's{seed}-{d}'] = {
                'nearest_neighbor_boundary_certificate_tolerance': 1e-12,
                'minimum_selected_minus_unselected_cosine': minimum_gap,
                'adjacency_conjugation_exact': True}
            q = baseline.T if d == 'T2V' else baseline
            for operator, beta in [('smooth', .5), ('sharp', -.5)]:
                for kind, current_graph in [('real', graph), ('shuffle', shuffled)]:
                    name = f's{seed}-{operator}-{kind}'
                    reconstructed = scalar_candidates(q, current_graph, beta)
                    s = reconstructed.T.copy() if d == 'T2V' else reconstructed
                    saved = load(parent['conditions'][name]['scores'][d])
                    assert np.array_equal(s, saved)
                    all_scores[name, d] = saved
    for operator in ('smooth', 'sharp'):
        for kind in ('real', 'shuffle'):
            name = f'ensemble-{operator}-{kind}'
            for d in ('T2V', 'V2T'):
                total = np.zeros((519, 519), dtype=np.float64)
                for seed in SEEDS:
                    total += all_scores[f's{seed}-{operator}-{kind}', d].astype(np.float64)
                expected = (total / 3).astype(np.float32)
                saved = load(parent['conditions'][name]['scores'][d])
                assert np.array_equal(expected, saved)
                all_scores[name, d] = saved
    for name, condition in parent['conditions'].items():
        for d in ('T2V', 'V2T'):
            s = all_scores[name, d]
            m = evaluate_score_matrix(s, video_ids=ids, text_ids=ids,
                                      video_to_text=positives, text_to_video=positives)
            original_m = condition['metrics'][d]
            assert all(m[d][f'R{k}'] == original_m[f'R{k}'] for k in (1, 5, 10))
            assert ranks(s)[d].tolist() == original_m['ranks']
            report['conditions'][f'{name}-{d}'] = {'scores_exact': True,
                                                    'official_R1_R5_R10_exact': True,
                                                    'diagnostic_ranks_exact': True}
        assert condition['metrics']['mean_R1'] == (
            condition['metrics']['T2V']['R1'] + condition['metrics']['V2T']['R1']) / 2
    assert len(report['conditions']) == 32 and len(report['graphs']) == 6
    report.update(status='completed', wall_seconds=time.time()-started, verified_score_matrices=32,
                  verified_official_recall_values=96, method_go=False)
    dump(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('conditions', 'graphs')}, indent=2))


if __name__ == '__main__':
    main()
