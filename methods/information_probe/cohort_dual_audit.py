"""Cohort-only assignment dual/tie audit, not independent retrieval inference."""
import json
import os
import subprocess
import time
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix

from .common import ART, ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def assignment(scores):
    i, p = linear_sum_assignment(scores, maximize=True)
    assert np.array_equal(i, np.arange(len(scores)))
    return p


def dual_certificate(scores):
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all():
        raise ValueError('finite square score matrix required')
    n = len(s)
    i, j = np.indices(s.shape)
    row = np.arange(n*n)
    constraints = coo_matrix((-np.ones(2*n*n), (np.concatenate((row, row)),
                            np.concatenate((i.ravel(), n+j.ravel())))), shape=(n*n, 2*n)).tocsr()
    bounds = [(None, None)]*(2*n)
    bounds[0] = (0., 0.)
    solved = linprog(np.ones(2*n), A_ub=constraints, b_ub=-s.ravel(), bounds=bounds,
                     method='highs', options={'time_limit': 120., 'dual_feasibility_tolerance': 1e-9,
                                              'primal_feasibility_tolerance': 1e-9})
    if not solved.success:
        raise RuntimeError(f'dual LP: {solved.status}: {solved.message}')
    u, v = solved.x[:n], solved.x[n:]
    p = assignment(s)
    reduced = s-u[:, None]-v[None, :]
    selected = reduced[np.arange(n), p]
    others = reduced.copy()
    others[np.arange(n), p] = -np.inf
    row_margin = selected-others.max(1)
    col_margin = selected-others.max(0)[p]
    primal = float(s[np.arange(n), p].sum())
    report = {'primal_objective': primal, 'dual_objective': float(solved.fun),
              'duality_gap': float(solved.fun-primal),
              'max_constraint_violation': float(max(0., reduced.max())),
              'max_selected_edge_abs_residual': float(np.abs(selected).max()),
              'row_min_assigned_margin': float(row_margin.min()),
              'column_min_assigned_margin': float(col_margin.min()),
              'row_assigned_nonstrict_max_fraction': float(np.mean(row_margin >= -1e-7)),
              'column_assigned_nonstrict_max_fraction': float(np.mean(col_margin >= -1e-7)),
              'row_assigned_strict_margin_fraction': float(np.mean(row_margin > 1e-7)),
              'column_assigned_strict_margin_fraction': float(np.mean(col_margin > 1e-7)),
              'solver_iterations': int(solved.nit)}
    assert abs(report['duality_gap']) <= 1e-7
    assert report['max_constraint_violation'] <= 1e-7
    assert report['max_selected_edge_abs_residual'] <= 1e-7
    assert row_margin.min() >= -1e-7 and col_margin.min() >= -1e-7
    return report, u, v, p


def noisy_assignments(scores, seeds=range(20), amplitude=2e-5):
    s = np.asarray(scores, dtype=np.float64)
    original = assignment(s)
    best = s[np.arange(len(s)), original].sum()
    results = []
    for seed in seeds:
        noise = np.random.default_rng(seed).uniform(-amplitude, amplitude, size=s.shape)
        p = assignment(s+noise)
        regret = float(best-s[np.arange(len(s)), p].sum())
        assert regret >= -1e-7 and regret <= 2*len(s)*amplitude+1e-7
        results.append({'noise_seed': seed, 'assignment': p.tolist(),
                        'changed_assignment_rows': int(np.sum(p != original)),
                        'original_score_regret': regret,
                        'promoted_full_gallery_R1_both_directions': float(100*np.mean(p == np.arange(len(s))))})
    return results


def main():
    path = OUT/'AS-C15-COHORT-AUDIT_run.json'
    if path.exists():
        raise FileExistsError(path)
    start = time.time()
    report = {'experiment_id': 'AS-C15-COHORT-AUDIT', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C15_protocol.md'),
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'fit_label_access': 'none; paired identity used only after solving to evaluate R1',
              'resource_contract': 'complete evaluation query cohort; NOT independent-query inference',
              'method_go': False, 'selector': None, 'noise_amplitude': 2e-5, 'seeds': {}}
    dump(path, report)
    try:
        previous = json.loads((OUT/'AS-C07-COHORT-V2.json').read_text())
        assert previous['status'] == 'completed'
        report['previous_run_sha256'] = sha(OUT/'AS-C07-COHORT-V2.json')
        for seed in (42, 1337, 2026):
            source = ROOT/f'runs/ph_base_b512_s{seed}/evaluation/dev/scores_video_x_text.npy'
            s = np.load(source).astype(np.float64)
            print(json.dumps({'seed': seed, 'stage': 'dual solve'}), flush=True)
            dual, u, v, p = dual_certificate(s)
            r1 = float(100*np.mean(p == np.arange(len(s))))
            reference = next(x for x in previous['results']['full_cohort']['seeds'] if x['seed'] == seed)
            assert abs(r1-reference['mean_R1']) < 1e-12
            noisy = noisy_assignments(s)
            values = np.array([x['promoted_full_gallery_R1_both_directions'] for x in noisy])
            baseline = next(x['mean_R1'] for x in previous['results']['independent_reference']['seeds'] if x['seed'] == seed)
            entry = {'score_sha256': sha(source), 'n': len(s), 'dual': dual,
                     'original_cohort_R1': r1, 'baseline_mean_R1': baseline, 'perturbations': noisy,
                     'noise_summary': {'mean_R1': float(values.mean()), 'min_R1': float(values.min()),
                                       'max_R1': float(values.max()), 'mean_gain_pp': float(values.mean()-baseline),
                                       'min_gain_pp': float(values.min()-baseline),
                                       'max_changed_assignment_rows': max(x['changed_assignment_rows'] for x in noisy)}}
            np.savez(ART/f'AS-C15-s{seed}-DEV_COHORT_ONLY.npz', u=u, v=v, assignment=p)
            report['seeds'][str(seed)] = entry
            dump(path, report)
            print(json.dumps({'seed': seed, 'dual': dual, 'noise_summary': entry['noise_summary']}), flush=True)
        report.update(status='completed', wall_seconds=time.time()-start,
                      limits=['Dual offsets are cohort-dependent and only certify non-strict assigned maxima.',
                              'Independent argmax tie-breaking is not certified to reproduce assignment.',
                              'Numerical stability is not linguistic identifiability or validation of one-to-one capacity.',
                              'Repeated jitter realizations are not independent training seeds or a GO pilot.'])
        dump(path, report)
    except Exception:
        report.update(status='failed', wall_seconds=time.time()-start, traceback=traceback.format_exc())
        dump(path, report)
        raise


if __name__ == '__main__':
    main()
