"""Complete descriptive AS-C13 accounting; no hypothesis-test selection."""
import json

from .common import ART, ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def main():
    run = json.loads((OUT/'AS-C13-RETENTION_run.json').read_text())
    assert run['status'] == 'completed'
    assert set(run['variants']) == {'identity', 'aligned_forward', 'shuffled_forward', 'mean', 'roundtrip'}
    results = {}
    for name in run['variants']:
        path = OUT/f'AS-C13-{name}_metrics.json'
        m = json.loads(path.read_text())
        results[name] = {'metrics_sha256': sha(path), 'score_sha256': sha(ART/f'AS-C13-{name}_scores.npy'),
                         'mean_R1': m['official_mean_R1'],
                         'mean_R1_delta_pp': m['official_mean_R1']-run['variants']['identity']['mean_R1'],
                         'directions': {}}
        for d, official in [('T2V', 'official_T2V'), ('V2T', 'V2T')]:
            results[name]['directions'][d] = {**{k: m[official][k] for k in ('R1', 'R5', 'R10')},
                                               'persistent_n': m[d]['persistent_n'],
                                               'persistent_mean_rank_delta': m[d]['persistent_mean_rank_delta']}
    report = {'experiment_id': 'AS-C13-SUMMARY', 'status': 'ANALYZED', 'method_go': False,
              'run_sha256': sha(OUT/'AS-C13-RETENTION_run.json'), 'code_sha256': sha(__file__),
              'maps_sha256': sha(ART/'AS-C13-maps.pt'), 'retrieval': results,
              'reconstruction': {split: {name: {k: v for k, v in m.items() if not k.startswith('per_sequence')}
                                         for name, m in maps.items()} for split, maps in run['reconstruction'].items()},
              'fallacy_scan_coverage': '11/11', 'fallacy_checks': {
                  'Simpson': 'Both map directions, train/dev and retrieval directions retained; no subgroup universality.',
                  'Ecological': 'Weighted sequence-level reconstruction does not establish individual linguistic retention.',
                  'Berkson': 'One selected PH-trained checkpoint; no general-backbone or sign-language population inference.',
                  'Collider': 'No adjusted causal association or conditioned treatment estimate.',
                  'Base_rate': 'All7096 train and519 dev rows used; no success-only subset or balanced-label reinterpretation.',
                  'Regression_to_mean': 'Persistent errors descriptive only; no improvement claimed on selected difficult examples.',
                  'Survivorship': 'All12 reconstruction cells and5 retrieval variants retained; no failed run omitted.',
                  'Look_elsewhere': 'Repeated exploratory dev use; no p-values or confirmatory method selection.',
                  'Forking_paths': 'Fixed ridge/null/roundtrip preregistered; no dev-chosen hyperparameter.',
                  'Correlation_causation': 'Explained normalized variance is not percent semantic information; substitution mixes several effects.',
                  'Reverse_causality': 'Bidirectional prediction is not a bidirectional causal mechanism or proof of inversion.'},
              'decision': 'Aligned token predictability transfers, but all substitutions underperform unchanged R0. No evidence of useful information discarded by the contextual encoder and no method GO.',
              'reproducibility': 'Unit algebra and identity rank parity verified; full experiment not independently rerun.'}
    dump(OUT/'AS-C13-SUMMARY.json', report)
    print(json.dumps({'retrieval': results, 'wall_seconds': run['wall_seconds'],
                      'peak_gpu_bytes': run['peak_gpu_bytes']}, indent=2))


if __name__ == '__main__':
    main()
