"""Deterministic stratified PH dev audit; no invented linguistic judgments."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import ART,ROOT,dump,persistent,ranks,rows,sha
from .scoring import pooled


def main():
    torch.set_num_threads(4)
    records=rows('dev')
    training=rows('train')
    native_rows=[json.loads(s) for s in (ROOT/'artifacts/proposal7/forensics/ph_dev.jsonl').read_text().splitlines()]
    native={r['id']:r for r in native_rows}
    cache=torch.load(ART/'frozen_dev.pt',weights_only=True)
    scores=np.load(ART/'baseline_dev_scores.npy')
    ranks0=ranks(scores)
    persistent_mask=persistent()
    vc=F.normalize(pooled(cache['video_tokens'],cache['video_mask']==0),dim=-1)
    tc=F.normalize(pooled(cache['text_tokens'],cache['text_mask']==1),dim=-1)
    dev_counts=Counter(r['caption_model'] for r in records)
    train_counts=Counter(r['caption_model'] for r in training)
    strata=defaultdict(list)
    for d in ('T2V','V2T'):
        for i in np.flatnonzero(persistent_mask[d]):
            key=(d,'top10' if ranks0[d][i]<10 else 'outside_top10',
                 'repeated' if dev_counts[records[i]['caption_model']]>1 else 'unique')
            strata[key].append(int(i))
    sample=[]
    for key,indexes in sorted(strata.items()):
        selected=sorted(indexes,key=lambda i:hashlib.sha256(records[i]['pair_id'].encode()).hexdigest())[:4]
        d=key[0]
        for i in selected:
            s=scores[:,i].copy() if d=='T2V' else scores[i].copy()
            s[i]=-np.inf
            competitors=np.argsort(-s,kind='stable')[:3]
            j=int(competitors[0])
            r,other=records[i],records[j]
            m=json.loads(Path(r['temporal_metadata']).read_text())
            om=json.loads(Path(other['temporal_metadata']).read_text())
            a,b=native[r['pair_id']],native[other['pair_id']]
            same_caption=r['caption_model']==other['caption_model']
            item={
                'stratum':list(key),'query_index':i,'direction':d,'pair_id':r['pair_id'],
                'baseline_rank':int(ranks0[d][i])+1,'strongest_confuser_id':other['pair_id'],
                'caption_original':r['caption_original'],'caption_model':r['caption_model'],
                'confuser_caption_original':other['caption_original'],'confuser_caption_model':other['caption_model'],
                'duration_seconds':m['decoded_frame_count']/m['fps'],
                'confuser_duration_seconds':om['decoded_frame_count']/om['fps'],
                'correct_score':float(scores[i,i]),'confuser_score':float(s[j]),
                'correct_minus_confuser':float(scores[i,i]-s[j]),
                'text_feature_cosine':float(tc[i]@tc[j]),'video_feature_cosine':float(vc[i]@vc[j]),
                'source':a['source'],'confuser_source':b['source'],'same_source':a['source']==b['source'],
                'signer':a['signer'],'confuser_signer':b['signer'],'same_signer':a['signer']==b['signer'],
                'dev_exact_caption_count':dev_counts[r['caption_model']],
                'train_exact_caption_count':train_counts[r['caption_model']],
                'neighbors':[{'id':records[k]['pair_id'],'caption':records[k]['caption_model'],'score':float(s[k])}
                             for k in competitors],
                'OBSERVED':['persistent baseline error','correct/competitor scores and captions recorded']+
                           (['identical model-caption string'] if same_caption else []),
                'INFERRED':[],
                'REQUIRES_SIGN_LANGUAGE_EXPERT':['whether either caption is a valid description of either signed clip',
                                               'which manual/non-manual or temporal contrast, if any, disambiguates them'],
                'failure_category':'TEXT_COLLAPSE' if same_caption else 'UNKNOWN',
                'category_limit':'identical string is observed; no claim of semantic equivalence or irreducibility',
            }
            sample.append(item)
    result={'experiment_id':'AS-C01-ERROR-AUDIT','status':'measured_descriptive',
            'code_sha256':sha(__file__),'manifest_sha256':cache['manifest_sha256'],
            'checkpoint_sha256':cache['checkpoint_sha256'],'selection':'up to four hash-smallest IDs per fixed stratum',
            'stratum_populations':{'|'.join(k):len(v) for k,v in strata.items()},
            'n_sample':len(sample),'sample':sample,
            'limits':['stratified, not prevalence-weighted; not a random sample for unweighted prevalence claims',
                      'source/signers read from existing native annotation forensics',
                      'no expert linguistic review performed; no changed positives or dev training labels']}
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/AS-C01-ERROR-AUDIT.json',result)
    print(json.dumps({'n_sample':len(sample),'stratum_populations':result['stratum_populations']}))


if __name__=='__main__':
    main()
