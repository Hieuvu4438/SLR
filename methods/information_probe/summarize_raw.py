"""Full crossed raw-screen accounting; no partial-run or selected-seed claims."""
import csv
import io
import json

import numpy as np

from .common import ART,ROOT,dump,metrics,ranks,rows,sha
from .summarize_readout import cluster_bootstrap

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def main():
    seeds=(42,1337,2026)
    arms=('existing','pooled','spatial','shuffled','zero')
    records=rows('dev')
    labels=[r['video_id'].rsplit('-',1)[0] for r in records]
    baseline=np.load(ART/'baseline_dev_scores.npy')
    result={'experiment_id':'AS-C02-CROSSED-SUMMARY','status':'analyzed',
            'code_sha256':sha(__file__),'regimes':{},'rows':[]}
    for regime in ('R2','R3'):
        scores,ms,rs={},{},{}
        for arm in arms:
            scores[arm],ms[arm],rs[arm]=[],[],[]
            for seed in seeds:
                expid=f'AS-C02-{regime}-{arm}-s{seed}'
                report=json.loads((OUT/f'{expid}_run.json').read_text())
                assert report['status']=='completed',f'{expid} incomplete'
                s=np.load(ART/expid/'selected_scores.npy')
                m=metrics(s,baseline)
                scores[arm].append(s);ms[arm].append(m);rs[arm].append(report)
                result['rows'].append({'regime':regime,'arm':arm,'seed':seed,
                    'selected_step':report['selected_step'],'mean_R1':m['official_mean_R1'],
                    'T2V_R1':m['official_T2V']['R1'],'V2T_R1':m['V2T']['R1'],
                    'T2V_R5':m['official_T2V']['R5'],'V2T_R5':m['V2T']['R5'],
                    'T2V_R10':m['official_T2V']['R10'],'V2T_R10':m['V2T']['R10'],
                    'T2V_hard_pair':m['T2V']['hard_pair_accuracy'],'V2T_hard_pair':m['V2T']['hard_pair_accuracy'],
                    'parameters':report['parameters'],'wall_seconds':report['wall_seconds'],
                    'learned_gain_pp':report['learned_gain_pp']})
        means={arm:float(np.mean([m['official_mean_R1'] for m in ms[arm]])) for arm in arms}
        counts={r['parameters'] for arm in arms for r in rs[arm]}
        assert len(counts)==1,'parameter matching failed'
        control=max((a for a in arms if a!='spatial'),key=lambda a:means[a])
        gates=[];deltas=[]
        for i,seed in enumerate(seeds):
            m,c=ms['spatial'][i],ms[control][i]
            mr,cr=ranks(scores['spatial'][i]),ranks(scores[control][i])
            deltas.append(sum((mr[d]==0).astype(float)-(cr[d]==0) for d in ('T2V','V2T'))*50)
            gate={'seed':seed,'mean_gain_pp':m['official_mean_R1']-c['official_mean_R1'],
                  'gain_at_least_half_point':m['official_mean_R1']-c['official_mean_R1']>=.5,
                  'direction_nonregression':all(m[d]['R1']-c[d]['R1']>=-.25 for d in ('official_T2V','V2T')),
                  'R5_R10_nonregression':all(m[d][f'R{k}']-c[d][f'R{k}']>=-.5
                                           for d in ('official_T2V','V2T') for k in (5,10)),
                  'persistent_improvement':all(m[d]['persistent_mean_rank_delta']<c[d]['persistent_mean_rank_delta']
                                               for d in ('T2V','V2T')),
                  'gain_beyond_init':rs['spatial'][i]['learned_gain_pp']>0,
                  'tie_contract':m['official_T2V']['rank_entries']==c['official_T2V']['rank_entries']==519}
            gate['all_gates']=all(v for k,v in gate.items() if k not in ('seed','mean_gain_pp'))
            gates.append(gate)
        ci=cluster_bootstrap(np.mean(deltas,axis=0),labels)
        result['regimes'][regime]={'means':means,'matched_parameter_count':next(iter(counts)),
                'strongest_control':control,'per_seed':gates,'bootstrap':ci,
                'passed':sum(g['all_gates'] for g in gates)>=2 and ci['lower']>0}
    result['limits']=[
        'These are limited empirical readouts, not formal information-theoretic upper bounds.',
        'A negative result cannot classify irreducibility or prove frozen/raw information absent.',
        'Three readout optimization seeds share one baseline checkpoint.',
        'Clean-only screening omits historical word swaps equally across arms.',
        'Spatial must beat pooled, existing-token, shuffled and zero controls; pooled gains alone do not prove missing spatial information.',
        'Confidence intervals condition on a fixed dev gallery and inferred source-prefix clusters.',
        'Exploratory multi-cycle dev selection requires independent confirmation before generalization claims.']
    prior=json.loads((OUT/'AS-C01-R1_summary.json').read_text())
    result['fallacy_scan']=prior['fallacy_scan']
    result['fallacy_scan']['survivorship']='NOTE: all 30 registered crossed runs required; failures prevent summary'
    result['fallacy_scan']['correlation_causation']='CAUTION: matched information interventions identify only this extractor/readout; no linguistic localization without follow-up'
    dump(OUT/'AS-C02-CROSSED-SUMMARY.json',result)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(result['rows'][0]))
    writer.writeheader();writer.writerows(result['rows'])
    (OUT/'AS-C02-CROSSED-SUMMARY.csv').write_text(stream.getvalue())
    print(json.dumps(result['regimes'],indent=2))


if __name__=='__main__':
    main()
