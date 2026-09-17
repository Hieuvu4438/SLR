"""Train-only convex channel-mixture diagnostic; not a novel method.

Tests direction-specific calibration versus shared and shuffled-direction fitting.
No candidate bias, gallery pruning, changed positives, or dev-fitted weights.
"""
import json
import os
import subprocess
import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.nn import functional as F

from .common import ART,ROOT,dump,metrics,ranks,sha

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def paired(v,t,vm,tm,scale):
    a=torch.einsum('ifd,ild->ifl',F.normalize(v,dim=-1),F.normalize(t,dim=-1))
    vv,tv=vm==0,tm==1
    x=(a*(a/.07).softmax(-1)).sum(-1)
    y=(a*(a/.07).softmax(-2)).sum(-2)
    return torch.stack((scale*(x*vv).sum(-1)/vv.sum(-1),
                        scale*(y*tv).sum(-1)/tv.sum(-1)),-1)


@torch.inference_mode()
def margins():
    x=torch.load(ART/'frozen_train.pt',weights_only=True)
    hard=torch.load(ART/'train_confusers.pt',weights_only=True)
    assert x['manifest_sha256']==hard['manifest_sha256']
    v,t,vm,tm=(x[k].cuda() for k in ('video_tokens','text_tokens','video_mask','text_mask'))
    result={}
    for d,key in [('V2T','video_confuser'),('T2V','text_confuser')]:
        values=[]
        for i in range(0,len(v),128):
            idx=torch.arange(i,min(i+128,len(v)),device='cuda')
            other=hard[key][i:i+128].cuda()
            pos=paired(v[idx],t[idx],vm[idx],tm[idx],x['logit_scale'])
            vi,ti=(idx,other) if d=='V2T' else (other,idx)
            neg=paired(v[vi],t[ti],vm[vi],tm[ti],x['logit_scale'])
            values.append((pos-neg).cpu().numpy())
        result[d]=np.concatenate(values).astype(np.float64)
    return result,x


def fit(delta):
    def objective(eta):
        return float(np.logaddexp(0,-(eta*delta[:,0]+(1-eta)*delta[:,1])).mean())
    opt=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-10})
    assert opt.success
    candidates=[(objective(z),z) for z in (0.,float(opt.x),1.)]
    loss,eta=min(candidates)
    return {'eta':eta,'pair_logistic_loss':loss,'initial_loss':objective(.5),
            'n_train_pairs':len(delta),'function_evaluations':int(opt.nfev)+2}


def main():
    torch.set_num_threads(4)
    start=time.time()
    plan={'experiment_id':'AS-C03-DIRECTION','status':'running','pid':os.getpid(),
          'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'code_sha256':sha(__file__),'seed':42,'fit_split':'train only; 7096 strongest-confuser pairs per direction',
          'optimizer':'bounded scalar minimization, xatol=1e-10, explicit endpoints',
          'objective':'mean softplus(-eta*delta_channel0-(1-eta)*delta_channel1)',
          'selector':'train objective only, no dev selection',
          'controls':'eta=.5; tied eta; two train-shuffled-direction eta values',
          'protocol':'same full dev gallery, separate directional scoring explicitly reported',
          'classification':'calibration diagnostic, not novelty claim',
          'resource_note':'short margin extraction overlaps raw acquisition; timings are not isolated benchmark timings'}
    dump(OUT/'AS-C03-DIRECTION_run.json',plan)
    data,cache=margins()
    for d,delta in data.items():
        np.save(ART/f'train_pair_channel_deltas_{d}.npy',delta)
    shared=fit(np.concatenate(list(data.values())))
    directional={d:fit(x) for d,x in data.items()}
    all_pairs=np.concatenate((data['T2V'],data['V2T']))
    rng=np.random.default_rng(42)
    order=rng.permutation(len(all_pairs))
    n=len(all_pairs)//2
    shuffled={'T2V':fit(all_pairs[order[:n]]),'V2T':fit(all_pairs[order[n:]])}
    variants={'baseline':{d:{'eta':.5} for d in data},
              'shared':{d:shared for d in data},'directional':directional,'shuffled_direction':shuffled}
    base=np.load(ART/'baseline_dev_scores.npy')
    ch=np.load(ART/'baseline_dev_channels.npy')
    output={}
    for name,params in variants.items():
        result={}
        direction_ranks={}
        for d,p in params.items():
            eta=p['eta']
            score=eta*ch[0]+(1-eta)*ch[1]
            m=metrics(score,base)
            result[d]=m[d]
            if d=='T2V':
                result['official_T2V']=m['official_T2V']
            direction_ranks[d]=ranks(score)[d]
            np.save(ART/f'AS-C03-{name}_{d}_scores.npy',score)
        result['official_mean_R1']=(result['official_T2V']['R1']+result['V2T']['R1'])/2
        result['direction_agreement']=float(np.mean((direction_ranks['T2V']==0)==(direction_ranks['V2T']==0)))
        result['weights']=params
        dump(OUT/f'AS-C03-{name}_metrics.json',result)
        output[name]={'weights':params,'T2V_R1':result['official_T2V']['R1'],
                      'V2T_R1':result['V2T']['R1'],'mean_R1':result['official_mean_R1'],
                      'persistent_mean_rank_delta':{d:result[d]['persistent_mean_rank_delta'] for d in data}}
    plan.update(status='completed',checkpoint_sha256=cache['checkpoint_sha256'],
                manifest_sha256=cache['manifest_sha256'],wall_seconds=time.time()-start,results=output,
                limits=['Convex deterministic fitting is not three independent training seeds.',
                        'A positive result would motivate localization, not establish a new method.',
                        'The loss only sees strongest R0 train confusers, not the full training objective.'])
    dump(OUT/'AS-C03-DIRECTION_run.json',plan)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
