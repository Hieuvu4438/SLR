"""Label-free global assignment diagnostic; NOT query-independent retrieval.

Uses one-to-one cohort capacity as a deliberately strong additional assumption.
No label is used to solve assignments; paired labels are used only by evaluation.
Promotes an assigned candidate then retains the remaining full-gallery ranking.
"""
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import ART,ROOT,dump,metrics,ranks,rows,sha
from .summarize_readout import cluster_bootstrap

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def promoted(scores,groups,direction):
    result=scores.copy()
    boost=float(scores.max()-scores.min()+1)
    for ids in groups:
        sub=scores[:,ids] if direction=='T2V' else scores[ids,:]
        vi,ti=linear_sum_assignment(sub,maximize=True)
        if direction=='T2V':
            result[vi,ids[ti]]+=boost
        else:
            result[ids[vi],ti]+=boost
    return result


def main():
    records=rows('dev')
    labels=[r['video_id'].rsplit('-',1)[0] for r in records]
    hashes=np.array([int.from_bytes(hashlib.sha256(r['pair_id'].encode()).digest()[:4],'little') for r in records])
    groups={'full_cohort':[np.arange(519)],
            'half_cohort':[np.flatnonzero(hashes%2==k) for k in range(2)],
            'single_query':[np.array([i]) for i in range(519)],
            'independent_reference':[]}
    result={'experiment_id':'AS-C07-COHORT-V2','status':'running','code_sha256':sha(__file__),
            'selection':'no fit or dev selection; fixed full/half/single-query cohorts',
            'assumption':'one-to-one assignment within query cohort, all519 candidates retained',
            'label_access':'none in assignment; paired IDs only in metric evaluation',
            'results':{}}
    dump(OUT/'AS-C07-COHORT-V2.json',result)
    for name,cohorts in groups.items():
        deltas=[];entries=[]
        for seed in (42,1337,2026):
            p=ROOT/f'runs/ph_base_b512_s{seed}/evaluation/dev/scores_video_x_text.npy'
            base=np.load(p);br=ranks(base)
            entry={'seed':seed,'scores_sha256':sha(p)}
            correctness={}
            for d in ('T2V','V2T'):
                score=promoted(base,cohorts,d)
                m=metrics(score,base)
                entry[d]=m[d]
                if d=='T2V':
                    entry['official_T2V']=m['official_T2V']
                correctness[d]=(ranks(score)[d]==0).astype(float)-(br[d]==0)
                np.save(ART/f'AS-C07-{name}-s{seed}-{d}.npy',score)
            entry['mean_R1']=(entry['official_T2V']['R1']+entry['V2T']['R1'])/2
            dump(OUT/f'AS-C07-{name}-s{seed}_metrics.json',entry)
            deltas.append((correctness['T2V']+correctness['V2T'])*50)
            entries.append({'seed':seed,'T2V_R1':entry['official_T2V']['R1'],
                            'V2T_R1':entry['V2T']['R1'],'mean_R1':entry['mean_R1']})
        result['results'][name]={'seeds':entries,'bootstrap_vs_baseline':cluster_bootstrap(np.mean(deltas,axis=0),labels)}
        print(json.dumps({'condition':name,**result['results'][name]}),flush=True)
    result['status']='completed'
    result['method_go']=False
    result['limits']=[
        'Global assignment consumes the complete query cohort and imposes one-to-one capacity.',
        'This is transductive and not a valid replacement for ordinary independent-query retrieval without a changed resource contract.',
        'Repeated captions violate identifiability even when their paired IDs admit a formal permutation.',
        'Promotion can break input-equivalent ties through cohort assignment; not evidence of linguistic disambiguation.',
        'single_query forces an argmax promotion and changes legacy tie behavior; independent_reference preserves the exact original score matrix.',
        'A gain would support investigating cohort structure, not establish a novel method or frozen-feature sufficiency for independent queries.',
        'Standard assignment is not claimed novel; no dataset-general contribution follows from PH-only results.']
    dump(OUT/'AS-C07-COHORT-V2.json',result)


if __name__=='__main__':
    main()
