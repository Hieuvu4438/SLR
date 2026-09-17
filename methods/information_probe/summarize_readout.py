"""Fixed diagnostic gates; paired source-prefix bootstrap conditional on gallery."""
import csv
import io
import json

import numpy as np

from .common import ART, ROOT, dump, metrics, ranks, rows, sha

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


def cluster_bootstrap(delta, labels, draws=10000):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    values = np.array([delta[labels == g].sum() for g in unique])
    sizes = np.array([(labels == g).sum() for g in unique])
    rng = np.random.default_rng(20260914)
    sample = rng.integers(len(unique), size=(draws, len(unique)))
    dist = values[sample].sum(1) / sizes[sample].sum(1)
    return {'mean_pp': float(delta.mean()), 'lower': float(np.quantile(dist, .025)),
            'upper': float(np.quantile(dist, .975)), 'draws': draws, 'clusters': len(unique),
            'bootstrap_seed': 20260914,
            'unit': 'filename prefix before final hyphen; inferred source, not verified recording identity',
            'scope': 'seed-averaged paired query correctness delta; conditional on fixed full dev gallery'}


def main():
    records = rows('dev')
    labels = [r['video_id'].rsplit('-', 1)[0] for r in records]
    baseline = np.load(ART/'baseline_dev_scores.npy')
    seeds = (42,1337,2026)
    modes = ('pooled','interaction','zero')
    values, measured, runs = {}, {}, {}
    summary_rows=[]
    for mode in modes:
        values[mode], measured[mode], runs[mode] = [], [], []
        for seed in seeds:
            expid=f'AS-C01-R1-{mode}-s{seed}'
            report=json.loads((OUT/f'{expid}_run.json').read_text())
            assert report['status']=='completed',f'{expid} not complete'
            score=np.load(ART/expid/'selected_scores.npy')
            m=metrics(score,baseline)
            values[mode].append(score)
            measured[mode].append(m)
            runs[mode].append(report)
            summary_rows.append({'mode':mode,'seed':seed,'step':report['selected_step'],
                                 'T2V_R1':m['official_T2V']['R1'],'V2T_R1':m['V2T']['R1'],
                                 'mean_R1':m['official_mean_R1'],'parameters':report['parameters'],
                                 'wall_seconds':report['wall_seconds'],
                                 'learned_gain_pp':report['learned_gain_pp']})
    means={mode:float(np.mean([m['official_mean_R1'] for m in measured[mode]])) for mode in modes}
    strongest=max(('pooled','zero'),key=lambda mode:means[mode])
    by_seed=[]
    deltas=[]
    for i,seed in enumerate(seeds):
        m,c=measured['interaction'][i],measured[strongest][i]
        mr,cr=ranks(values['interaction'][i]),ranks(values[strongest][i])
        delta=((mr['T2V']==0).astype(float)-(cr['T2V']==0)+
               (mr['V2T']==0).astype(float)-(cr['V2T']==0))*50
        deltas.append(delta)
        gate={
            'seed':seed,'mean_gain_pp':m['official_mean_R1']-c['official_mean_R1'],
            'mean_gain_at_least_half_point':m['official_mean_R1']-c['official_mean_R1']>=.5,
            'direction_nonregression':all(m['official_T2V' if d=='T2V' else d]['R1']-
                                         c['official_T2V' if d=='T2V' else d]['R1']>=-.25
                                         for d in ('T2V','V2T')),
            'R5_R10_nonregression':all(m[d][f'R{k}']-c[d][f'R{k}']>=-.5
                                     for d in ('official_T2V','V2T') for k in (5,10)),
            'persistent_mean_rank_improves':all(m[d]['persistent_mean_rank_delta']<
                                               c[d]['persistent_mean_rank_delta'] for d in ('T2V','V2T')),
            'learned_beyond_init':runs['interaction'][i]['learned_gain_pp']>0,
            't2v_tie_expansion_absent': m['official_T2V']['rank_entries']==c['official_T2V']['rank_entries']==519,
        }
        gate['all_per_seed_gates']=all(v for k,v in gate.items() if k not in ('seed','mean_gain_pp'))
        by_seed.append(gate)
    bootstrap=cluster_bootstrap(np.mean(deltas,axis=0),labels)
    result={
        'experiment_id':'AS-C01-R1-SUMMARY','status':'analyzed','means':means,
        'strongest_matched_control':strongest,'per_seed':by_seed,'bootstrap':bootstrap,
        'full_gate_pass':sum(x['all_per_seed_gates'] for x in by_seed)>=2 and bootstrap['lower']>0,
        'code_sha256':sha(__file__), 'summary':summary_rows,
        'caveats':[
            'Small screening readout cannot prove information absent if negative.',
            'Three probe seeds share one frozen backbone; no backbone-retraining variance estimate.',
            'Clean-only objective omits dynamic random swaps equally for all arms.',
            'Source prefix clusters are inferred. CI is not unconditional dataset generalization.',
            'Dev checkpoint selection and many diagnostics make this exploratory, not independent confirmation.',
            'R2/R3 and train-shuffled raw controls have not been run by this script.',
        ],
        'fallacy_scan':{
            'Simpson': 'CAUTION: direction and persistent strata reported; source-specific effects not fully identified',
            'ecological':'NOTE: claims limited to per-query rankings, not individual signer semantics',
            'Berkson':'CAUTION: persistent slice selected on baseline failures; full gallery primary',
            'collider':'NOTE: no fitted covariate adjustment; selected error strata not causal populations',
            'base_rate':'NOTE: full-gallery and persistent population counts reported',
            'regression_to_mean':'NOTE: frozen baseline, zero init and matched controls compared',
            'survivorship':'NOTE: all nine expected runs required, failures block aggregate',
            'look_elsewhere':'CAUTION: exploratory selection; all arms/checkpoints retained',
            'forking_paths':'CAUTION: fixed 600-step screen; later design changes need new IDs',
            'correlation_causation':'CAUTION: manipulation tests readout not sign-language meaning or missing information',
            'reverse_causality':'NOTE: fixed cached inputs precede interventions; no causal claim from observational slices',
        },
    }
    dump(OUT/'AS-C01-R1_summary.json',result)
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=list(summary_rows[0]))
    writer.writeheader();writer.writerows(summary_rows)
    (OUT/'AS-C01-R1_summary.csv').write_text(stream.getvalue())
    print(json.dumps({k:result[k] for k in ('means','strongest_matched_control','bootstrap','full_gate_pass')},indent=2))


if __name__=='__main__':
    main()
