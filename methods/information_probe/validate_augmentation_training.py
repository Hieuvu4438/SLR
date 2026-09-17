"""AS-C40 matched-training integrity and preregistered endpoint comparisons."""
import json
import time

import numpy as np
import yaml

from slr_common.evaluation.cico_eval import evaluate_score_matrix
from .common import ART, ROOT, dump, metrics, ranks, rows, sha
from .encoder_grid_probe import comparison
from .historical_training_replay import OUT, exact_compare


def matched_batches(a, b):
    fields = ('batch_number', 'pair_ids', 'clean_sha256', 'visual_sha256')
    return len(a) == len(b) and all(all(x[k] == y[k] for k in fields) for x, y in zip(a, b))


def run_name(seed, condition):
    # Registered infrastructure repair: original2026ON failed with BrokenPipe,
    # not an unfavorable endpoint. Its immutable record remains excluded/incomplete.
    suffix = '-attempt2' if (seed, condition) == (2026, 'on') else ''
    return f'AS-C40-s{seed}-{condition}{suffix}'


def main():
    target = OUT/'AS-C40-VALIDATION_run.json'
    if target.exists():
        raise FileExistsError(target)
    started = time.time()
    records = rows('dev')
    ids = [r['pair_id'] for r in records]
    labels = [r['video_id'].rsplit('-', 1)[0] for r in records]
    positives = {i: [i] for i in ids}
    strong = np.load(ART/'baseline_dev_scores.npy')
    ensemble = np.load(ART/'AS-C32-uniform3_scores.npy')
    result = {'experiment_id': 'AS-C40-VALIDATION', 'status': 'validating',
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C40_protocol.md'),
              'method_go': False, 'test_loaded': False, 'seeds': {}}
    # Refuse incomplete inputs before creating any validation output.
    runs = {}
    for seed in (42, 1337, 2026):
        runs[seed] = {}
        for condition in ('on', 'off'):
            path = OUT/f'{run_name(seed, condition)}_run.json'
            r = json.loads(path.read_text())
            assert r['status'] == 'completed' and r['updates'] == 260
            assert r['protocol_sha256'] == result['protocol_sha256']
            assert len(r['epochs']) == 20 and len(r['training_batches']) == 260
            runs[seed][condition] = r
    for seed, pair in runs.items():
        on, off = pair['on'], pair['off']
        configs = {c: yaml.safe_load((ART/f'{run_name(seed, c)}/resolved_config.yaml').read_text()) for c in pair}
        assert configs['on']['seed'] == configs['off']['seed'] == seed
        assert configs['on']['data']['text_augmentation'] == 'cico_random_swap_v1'
        assert configs['off']['data']['text_augmentation'] is None
        configs['off']['data']['text_augmentation'] = configs['on']['data']['text_augmentation']
        assert not exact_compare(configs['on'], configs['off'])
        assert on['initial_state_sha256'] == off['initial_state_sha256']
        assert on['source_checks'] == off['source_checks']
        assert on['parent_replay_sha256'] == off['parent_replay_sha256']
        assert on['code_sha256'] == off['code_sha256']
        assert on['endpoint_scheduler'] == off['endpoint_scheduler']
        assert matched_batches(on['training_batches'], off['training_batches'])
        assert all(x['changed_text_row_n'] == 0 and x['clean_sha256'] == x['aug_sha256'] for x in off['training_batches'])
        scores, initials, all_metrics, log_hashes = {}, {}, {}, {}
        for condition, r in pair.items():
            name = run_name(seed, condition)
            log_path = ART/f'{name}/train.jsonl'
            logs = [json.loads(x) for x in log_path.read_text().splitlines()]
            training = [x for x in logs if x.get('event') != 'dev_evaluation']
            evaluations = [x for x in logs if x.get('event') == 'dev_evaluation']
            assert len(training) == 260 and len(evaluations) == 20
            assert all(x['step'] == i+1 and x['epoch'] == i//13 for i, x in enumerate(training))
            assert all(x['forward_counts'] == {'student_text': 2, 'student_video': 1, 'teacher_text': 0, 'teacher_video': 0} for x in training)
            assert all(x['actual_global_negative_pool_per_query'] == 511 and x['effective_optimizer_batch'] == 512 for x in training)
            log_hashes[condition] = sha(log_path)
            path = ART/f'{name}-epoch19_scores.npy'
            assert sha(path) == r['epochs'][-1]['score_sha256']
            scores[condition] = np.load(path)
            ip = ART/f'{name}-initialization_scores.npy'
            assert sha(ip) == r['initialization']['score_sha256']
            initials[condition] = np.load(ip)
            ep = ART/f'{name}-epoch0_scores.npy'
            assert sha(ep) == r['epochs'][0]['score_sha256']
            for label, s in [('endpoint', scores[condition]), ('initialization', initials[condition]), ('epoch0', np.load(ep))]:
                m = metrics(s, strong)
                official = evaluate_score_matrix(s, video_ids=ids, text_ids=ids,
                                                video_to_text=positives, text_to_video=positives)
                for d, key in [('T2V', 'official_T2V'), ('V2T', 'V2T')]:
                    assert all(m[key][f'R{k}'] == official[d][f'R{k}'] for k in (1, 5, 10))
                    if label != 'initialization':
                        epoch = 19 if label == 'endpoint' else 0
                        assert all(m[key][f'R{k}'] == r['epochs'][epoch]['metrics'][d][f'R{k}'] == evaluations[epoch]['metrics'][d][f'R{k}'] for k in (1, 5, 10))
                    else:
                        assert m[key]['R1'] == r['initialization']['R1'][d]
                all_metrics[f'{condition}_{label}'] = m
        assert np.array_equal(initials['on'], initials['off'])
        m, b = metrics(scores['off'], scores['on']), metrics(scores['on'])
        ci = comparison(scores['off'], scores['on'], labels, f'seed{seed}OFF minus ON at260updates')
        gate = {'mean_gain_at_least_half_pp': ci['mean_pp'] >= .5, 'bootstrap_lower_positive': ci['lower'] > 0,
                'R1_nonregression': all(m[d]['R1']-b[d]['R1'] >= -.25 for d in ('official_T2V', 'V2T')),
                'R5_R10_nonregression': all(m[d][f'R{k}']-b[d][f'R{k}'] >= -.5 for d in ('official_T2V', 'V2T') for k in (5, 10)),
                'persistent_ranks_improve_both': all(m[d]['persistent_mean_rank_delta'] < 0 for d in ('T2V', 'V2T'))}
        gate['training_harm_diagnostic_lead'] = all(gate.values())
        result['seeds'][str(seed)] = {
            'run_hashes': {c: sha(OUT/f'{run_name(seed, c)}_run.json') for c in pair},
            'initial_state_exact': True, 'batch_order_clean_visual_exact': True,
            'configs_differ_only_augmentation': True, 'training_log_sha256': log_hashes,
            'all260_forward_counts_and_negative_pool_exact': True,
            'initialization_scores_exact': True, 'scheduler_exact': True,
            'training_row_occurrences': sum(len(x['pair_ids']) for x in on['training_batches']),
            'on_augmented_occurrences': sum(x['changed_text_row_n'] for x in on['training_batches']),
            'off_augmented_occurrences': 0, 'metrics': all_metrics, 'off_minus_on': m,
            'bootstrap_off_minus_on': ci,
            'bootstrap_off_minus_initialization': comparison(scores['off'], initials['off'], labels, f'seed{seed}OFF endpoint minus own initialization'),
            'bootstrap_off_minus_strong42': comparison(scores['off'], strong, labels, f'seed{seed}OFF endpoint minus original strong42'),
            'bootstrap_off_minus_ensemble': comparison(scores['off'], ensemble, labels, f'seed{seed}OFF endpoint minus AS-C32ensemble'),
            'gate': gate}
    result['passing_seed_n'] = sum(s['gate']['training_harm_diagnostic_lead'] for s in result['seeds'].values())
    result.update(status='completed', training_harm_diagnostic_lead=result['passing_seed_n'] >= 2,
                  wall_seconds=time.time()-started, official_validated_matrix_n=18, independent_method_go=False)
    dump(target, result)
    print(json.dumps({'status': result['status'], 'passing_seed_n': result['passing_seed_n'],
                      'seeds': {k: {'ci': v['bootstrap_off_minus_on'], 'gate': v['gate']} for k, v in result['seeds'].items()}}, indent=2))


if __name__ == '__main__':
    main()
