"""AS-C17 accounting, including post-hoc effective view-diversity adequacy."""
from collections import Counter
import json

import numpy as np

from slr_common.data.views import jittered_view
from .common import ART, ROOT, dump, ranks, rows, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def main():
    path = OUT/'AS-C17-VIEWS_run.json'
    run = json.loads(path.read_text())
    assert run['status'] == 'completed'
    records = rows('dev')
    diversity = []
    for row in records:
        keys = {tuple(jittered_view(row['dense_length'], 64, global_seed=s, video_id=row['video_id'])[0].indices)
                for s in (42, 1337, 2026)}
        diversity.append(len(keys))
    all_counts = Counter(diversity)
    long_counts = Counter(x for x, r in zip(diversity, records) if r['dense_length'] > 64)
    singles = [np.load(ART/f'AS-C17-jitter_{s}_scores.npy') for s in (42, 1337, 2026)]
    same_rows = np.all(singles[0] == singles[1], axis=1)&np.all(singles[0] == singles[2], axis=1)
    assert np.array_equal(same_rows, np.array(diversity) == 1)
    control = np.load(ART/'AS-C17-canonical_mean3_scores.npy')
    average = np.load(ART/'AS-C17-jitter_mean3_scores.npy')
    unchanged = np.array([r['dense_length'] <= 64 for r in records])
    assert np.array_equal(control[unchanged], average[unchanged])
    cr, ar = ranks(control), ranks(average)
    result = {'experiment_id': 'AS-C17-SUMMARY', 'status': 'ANALYZED', 'method_go': False,
              'run_sha256': sha(path), 'code_sha256': sha(__file__),
              'posthoc_disclosure': 'After the fixed screen, exact cross-seed view diversity and unchanged-video score-row checks added to qualify probe adequacy; no revised sampler or rerun.',
              'unique_views_per_video': diversity, 'unique_view_counts_all': dict(all_counts),
              'unique_view_counts_long_only': dict(long_counts),
              'identical_score_rows_across_three_views': int(same_rows.sum()),
              'unchanged_video_n': int(unchanged.sum()),
              'unchanged_video_V2T_score_row_parity': True,
              'rank_changes': {d: {'n': int(np.count_nonzero(ar[d]-cr[d])),
                                   'newly_correct': int(np.sum((cr[d] > 0)&(ar[d] == 0))),
                                   'newly_wrong': int(np.sum((cr[d] == 0)&(ar[d] > 0))),
                                   'unchanged_own_video_query_rank_changes': int(np.count_nonzero((ar[d]-cr[d])[unchanged]))}
                               for d in cr},
              'variant_deltas_pp': {k: v['mean_R1']-run['variants']['canonical_mean3']['mean_R1'] for k, v in run['variants'].items()},
              'fallacy_scan_coverage': '11/11', 'fallacy_checks': {
                  'Simpson': 'Overall R1 losses and persistent mean-rank improvements both reported; no all-metrics-negative claim.',
                  'Ecological': 'Window sensitivity not assigned to specific linguistic cues or signers.',
                  'Berkson': 'One PH-selected checkpoint; narrow nearby-window perturbation, not all acquisition methods.',
                  'Collider': 'Length strata descriptive, not a causal duration adjustment.',
                  'Base_rate': '379 changed/140 unchanged videos and cross-seed diversity counts disclosed.',
                  'Regression_to_mean': 'No gain claimed from improved ranks on selected persistent errors.',
                  'Survivorship': 'All6 encoding passes and8 score outputs retained; no failure omitted.',
                  'Look_elsewhere': 'Fixed seeds/mean, no dev-selected jitter or significance claim.',
                  'Forking_paths': 'Protocol preceded run; diversity audit explicitly post-hoc; no rescue sampler.',
                  'Correlation_causation': 'Low-diversity jitter screen does not prove sampling is causally unimportant.',
                  'Reverse_causality': 'Score changes do not explain training adaptation; T2V gallery interference can affect unchanged-own-video queries.'},
              'decision': 'Fixed view mean fails R1 screen; effective diversity weakens general inference. No method or global sampling rejection.'}
    dump(OUT/'AS-C17-SUMMARY.json', result)
    print(json.dumps({k: result[k] for k in ('unique_view_counts_all', 'unique_view_counts_long_only', 'rank_changes')}, indent=2))


if __name__ == '__main__':
    main()
