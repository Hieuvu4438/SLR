"""Descriptive source/signer-conditioned confusions; never a causal label."""
from collections import Counter
import json

import numpy as np
import torch
from torch.nn import functional as F

from .common import ART,ROOT,dump,rows,ranks,persistent,sha
from .scoring import pooled


def main():
    torch.set_num_threads(4)
    records=rows('dev')
    path=ROOT/'artifacts/proposal7/forensics/ph_dev.jsonl'
    native={r['id']:r for r in map(json.loads,path.read_text().splitlines())}
    cache=torch.load(ART/'frozen_dev.pt',weights_only=True)
    assert cache['ids']==[r['pair_id'] for r in records]
    text=F.normalize(pooled(cache['text_tokens'],cache['text_mask']==1),dim=-1).numpy()
    similarity=text@text.T
    labels={key:np.array([native[r['pair_id']][key] for r in records]) for key in ('signer','source')}
    assert all(all(x not in ('',None,'unknown') for x in values) for values in labels.values())
    result={'experiment_id':'AS-C10-NUISANCE','status':'measured_descriptive',
            'code_sha256':sha(__file__),'manifest_sha256':cache['manifest_sha256'],
            'metadata_sha256':sha(path),'text_covariate_checkpoint_sha256':cache['checkpoint_sha256'],
            'metadata_counts':{k:dict(Counter(v.tolist())) for k,v in labels.items()},
            'control':'20 other candidates nearest in pooled-text cosine to the strongest confuser; no dev fitting',
            'seeds':{},'method_go':False}
    persistent_mask=persistent()
    for seed in (42,1337,2026):
        scores=np.load(ROOT/f'runs/ph_base_b512_s{seed}/evaluation/dev/scores_video_x_text.npy')
        baseline_rank=ranks(scores)
        seed_result={}
        for direction,axis in [('T2V',0),('V2T',1)]:
            wrong=scores.copy();np.fill_diagonal(wrong,-np.inf)
            confuser=wrong.argmax(axis=axis)
            masks={'all':np.ones(len(records),bool),'errors':baseline_rank[direction]>0,
                   'persistent_errors':persistent_mask[direction]}
            neighbors=[]
            for i,j in enumerate(confuser):
                diff=np.abs(similarity[i]-similarity[i,j])
                diff[[i,j]]=np.inf
                neighbors.append(np.argsort(diff,kind='stable')[:20])
            neighbors=np.array(neighbors)
            direction_result={}
            for stratum,mask in masks.items():
                entry={'n':int(mask.sum()),'covariate_mean_abs_cosine_gap':float(np.abs(
                    similarity[np.arange(len(records))[:,None],neighbors]-similarity[np.arange(len(records)),confuser,None])[mask].mean())}
                for key,values in labels.items():
                    same=values==values[confuser]
                    available=((values[:,None]==values[None,:]).sum(1)-1)/(len(values)-1)
                    matched=(values[:,None]==values[neighbors]).mean(1)
                    entry[key]={'observed_same_confuser_n':int(same[mask].sum()),
                                'observed_same_confuser_fraction':float(same[mask].mean()),
                                'uniform_gallery_expected_fraction':float(available[mask].mean()),
                                'text_cosine_matched_expected_fraction':float(matched[mask].mean()),
                                'descriptive_excess_vs_matched_pp':float(100*(same[mask]-matched[mask]).mean())}
                direction_result[stratum]=entry
            direction_result['signer_error_rates']={value:{'n':int((labels['signer']==value).sum()),
                'R1':float(100*(baseline_rank[direction][labels['signer']==value]==0).mean())}
                for value in sorted(set(labels['signer']))}
            seed_result[direction]=direction_result
        result['seeds'][str(seed)]=seed_result
    result['limits']=['Text cosine is a model-derived covariate, not verified semantic matching.',
                      'Source prefix is inferred recording group; native signer fields not independently reannotated.',
                      'Confuser association is not causal evidence of nuisance reliance.',
                      'No conditional independence, uncertainty interval, linguistic equivalence or population prevalence claim.',
                      'Metadata may correlate with content; no correction, new positives or new task is authorized.',
                      'Same seed42 text covariate used for all historical baselines; not independent nuisance models.']
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/AS-C10-NUISANCE.json',result)
    print(json.dumps({s:{d:r['persistent_errors'] for d,r in x.items()} for s,x in result['seeds'].items()},indent=2))


if __name__=='__main__':
    main()
