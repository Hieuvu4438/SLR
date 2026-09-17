"""Read-only historical learning-trajectory diagnosis; no new selection."""
import json
import subprocess
from collections import defaultdict

import numpy as np

from .common import ROOT, dump, sha, ranks as score_ranks


def main():
    result = {'experiment_id': 'AS-C09-TRAJECTORY', 'status': 'analyzed',
              'code_sha256': sha(__file__), 'new_training_updates': 0,
              'selector': 'none; report historical selectors and descriptive envelopes', 'seeds': {}}
    for seed in (42, 1337, 2026):
        run = ROOT/f'runs/ph_base_b512_s{seed}'
        records = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
        train, dev = defaultdict(list), []
        for row in records:
            if row.get('event') == 'dev_evaluation':
                dev.append(row)
            elif 'loss' in row:
                train[row['epoch']].append(row['loss'])
        selection = json.loads((run/'selection.json').read_text())
        provenance = json.loads((run/'provenance.json').read_text())
        revision = provenance.get('implementation', {}).get('commit')
        if not revision:
            revision = json.loads((run/'implementation_provenance.json').read_text())['implementation_git_commit']
        historical_eval = subprocess.check_output(['git', 'show', f'{revision}:elsc/evaluate.py'], text=True)
        historical_runtime = subprocess.check_output(['git', 'show', f'{revision}:elsc/evaluation/runtime.py'], text=True)
        assert 'shuffle=False' in historical_eval
        assert 'video_ids.append(video_id)' in historical_runtime and 'text_ids.append(text_id)' in historical_runtime
        saved_ranks = score_ranks(np.load(run/'evaluation/dev/scores_video_x_text.npy'))
        per_epoch = []
        for row in dev:
            m = row['metrics']
            per_epoch.append({'epoch': row['epoch'], 'step': row['step'],
                              'train_mean_loss': float(np.mean(train[row['epoch']])) if train.get(row['epoch']) else None,
                              'mean_R1': float(np.mean([m[d]['R1'] for d in ('T2V', 'V2T')])),
                              'T2V_R1': m['T2V']['R1'], 'V2T_R1': m['V2T']['R1']})
        best_idx = next(i for i, r in enumerate(dev) if r['epoch'] == selection['selected_epoch'])
        best = dev[best_idx]
        directions = {}
        for direction in ('T2V', 'V2T'):
            ranks = np.asarray([r['metrics'][direction]['cols'] for r in dev])
            assert ranks.shape == (len(dev), 519), 'tie-expanded counts prohibit per-query comparison'
            selected = ranks[best_idx]
            assert np.array_equal(selected, saved_ranks[direction]), 'logged selected per-query order does not match saved evaluation'
            hits = ranks == 0
            directions[direction] = {
                'selected_R1_from_ranks': float(100*hits[best_idx].mean()),
                'logged_selected_R1': best['metrics'][direction]['R1'],
                'last_R1': float(100*hits[-1].mean()),
                'selected_correct_lost_at_last_n': int((hits[best_idx]&~hits[-1]).sum()),
                'selected_wrong_corrected_at_last_n': int((~hits[best_idx]&hits[-1]).sum()),
                'selected_errors_ever_correct_in_trajectory_n': int((~hits[best_idx]&hits.any(0)).sum()),
                'always_wrong_n': int((~hits.any(0)).sum()),
                'ever_correct_R1_ORACLE_not_method': float(100*hits.any(0).mean()),
                'queries_changing_top1_correctness_n': int((hits.any(0)&~hits.all(0)).sum()),
                'selected_errors_median_correct_epoch_fraction': float(np.median(hits[:,selected>0].mean(0))),
                'mean_rank_change_last_vs_selected': float((ranks[-1]-selected).mean()),
            }
            assert abs(directions[direction]['selected_R1_from_ranks']-best['metrics'][direction]['R1']) < 1e-9
        entry = {'log_sha256': sha(run/'train.jsonl'), 'selection': selection,
                 'ordering_check': 'all519 selected ranks exactly match saved score ranks in both directions; pinned evaluator uses sequential first-occurrence IDs',
                 'implementation_revision': revision,
                 'dev_evaluations': len(dev), 'train_epochs': len(train),
                 'first': per_epoch[0], 'last': per_epoch[-1], 'historical_selected': per_epoch[best_idx],
                 'directions': directions, 'checkpoint_files': [
                     {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size}
                     for p in sorted((run/'checkpoints').glob('*.pt'))]}
        dump(ROOT/f'docs/proposal7/evidence/autonomous_search/AS-C09-s{seed}-trajectory.json', per_epoch)
        result['seeds'][str(seed)] = entry
    result['execution_notes'] = ['First invocation failed before report write: st_size integer called as function; fixed property access.',
                                 'Second invocation found initial epoch-1 evaluation with no training loss; report null instead of inventing loss. Seed42 trajectory had already been written and was regenerated unchanged.']
    result['limits'] = [
        'Descriptive reused dev trajectories; not independent confirmation or a new selector.',
        'Epoch0 means after the first13 updates, not pretraining or random initialization.',
        'Training loss contains dynamic augmentation and changing minibatches, not fixed full-gallery loss.',
        'Ever-correct envelope uses true labels per query and cannot be deployed.',
        'Only selected/last weights exist; intermediate rank logs cannot support representation-drift attribution.',
        'All baselines start from PH-fitted release weights; no clean training-from-scratch inference.',
        'No permission to reopen RPCA, generic gradient surgery, soup or early-stopping as novelty.']
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/AS-C09-TRAJECTORY.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
