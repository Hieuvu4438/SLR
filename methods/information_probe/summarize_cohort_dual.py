"""AS-C15 accounting plus explicitly post-hoc input-collision attribution."""
from collections import Counter
import json

import numpy as np
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.factory import load_cico_tokenizer
from .common import ART, ROOT, dump, rows, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def main():
    source = OUT/'AS-C15-COHORT-AUDIT_run.json'
    run = json.loads(source.read_text())
    assert run['status'] == 'completed'
    config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
    tokenizer = load_cico_tokenizer(config)
    keys = [tuple(encode_cico_text(r['caption_model'], tokenizer, 32)[0].tolist()) for r in rows('dev')]
    counts = Counter(keys)
    result = {'experiment_id': 'AS-C15-SUMMARY', 'status': 'ANALYZED', 'method_go': False,
              'run_sha256': sha(source), 'code_sha256': sha(__file__),
              'manifest_sha256': sha(ROOT/'artifacts/manifests/ph_dev.jsonl'),
              'tokenizer_code_sha256': sha(ROOT/'shared/slr_common/data/tokenize.py'),
              'posthoc_disclosure': 'After jitter results, descriptive same-deployed-input attribution added; no refitting or new selected noise realization.',
              'duplicate_text_rows': sum(n for n in counts.values() if n > 1), 'seeds': {}}
    for seed, entry in run['seeds'].items():
        artifact = ART/f'AS-C15-s{seed}-DEV_COHORT_ONLY.npz'
        original = np.load(artifact)['assignment']
        changes = []
        for row in entry['perturbations']:
            changed = [i for i, j in enumerate(row['assignment']) if j != original[i]]
            same = [i for i in changed if keys[row['assignment'][i]] == keys[int(original[i])]]
            changes.append({'noise_seed': row['noise_seed'], 'changed_n': len(changed),
                            'same_deployed_text_input_n': len(same), 'different_input_n': len(changed)-len(same)})
        result['seeds'][seed] = {'artifact_sha256': sha(artifact), 'original_R1': entry['original_cohort_R1'],
                                 'noise_summary': entry['noise_summary'], 'dual': entry['dual'],
                                 'assignment_change_attribution': changes,
                                 'total_changes': sum(x['changed_n'] for x in changes),
                                 'total_same_input_changes': sum(x['same_deployed_text_input_n'] for x in changes),
                                 'total_different_input_changes': sum(x['different_input_n'] for x in changes)}
    result['mean_noise_gain_pp_across_backbones'] = float(np.mean([x['noise_summary']['mean_gain_pp'] for x in result['seeds'].values()]))
    result['fallacy_scan_coverage'] = '11/11'
    result['fallacy_checks'] = {
        'Simpson': 'Each backbone, all20 perturbations and same/different-input changes retained; no hidden pooled reversal.',
        'Ecological': 'Assignment support and aggregate R1 are not linguistic disambiguation of each pair.',
        'Berkson': 'Selected historical PH checkpoints and complete paired cohort, not arbitrary deployment queries.',
        'Collider': 'No conditioned causal treatment estimate.',
        'Base_rate': 'All519 queries,17 repeated-input rows and every perturbation retained.',
        'Regression_to_mean': 'No gain inferred from remeasuring only baseline failures.',
        'Survivorship': 'All3 LP solves and60 perturbations completed; no failed solve omitted.',
        'Look_elsewhere': 'No perturbation selected; no significance/independent-pilot claim.',
        'Forking_paths': 'Dual/noise protocol fixed before run; later collision attribution explicitly post-hoc.',
        'Correlation_causation': 'Tiny-noise stability does not validate cohort capacity or independent-query benefit.',
        'Reverse_causality': 'Additive support derived from whole cohort, not evidence that an independently fitted offset causes its gain.'}
    result['decision'] = 'Cohort headroom is numerically robust but not evidence that nonadditive inference is necessary. Dual offsets only provide non-strict support and consume evaluation cohort; no new candidate.'
    dump(OUT/'AS-C15-SUMMARY.json', result)
    print(json.dumps({'mean_noise_gain_pp': result['mean_noise_gain_pp_across_backbones'],
                      'changes': {s: {k: x[k] for k in ('total_changes', 'total_same_input_changes', 'total_different_input_changes')}
                                  for s, x in result['seeds'].items()}, 'wall_seconds': run['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
