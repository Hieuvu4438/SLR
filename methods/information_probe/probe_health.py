"""Check train hard-pair support before interpreting a negative readout."""
import json

import torch

from .common import ART,ROOT,dump,sha
from .scoring import channels


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    x=torch.load(ART/'frozen_train.pt',weights_only=True)
    c=torch.load(ART/'train_confusers.pt',weights_only=True)
    v,t,vm,tm=(x[k].cuda() for k in ('video_tokens','text_tokens','video_mask','text_mask'))
    res={'experiment_id':'AS-C01-HEALTH','checkpoint_sha256':x['checkpoint_sha256'],
         'manifest_sha256':x['manifest_sha256'],'code_sha256':sha(__file__),
         'split':'train','candidate_n':len(v),'directions':{}}
    for d,key in [('V2T','video_confuser'),('T2V','text_confuser')]:
        margins=[]
        for i in range(0,len(v),64):
            idx=torch.arange(i,min(i+64,len(v)),device='cuda')
            other=c[key][i:i+64].cuda()
            a,b=channels(v[idx],t[idx],vm[idx],tm[idx],x['logit_scale'])
            pos=((a+b)/2).diag()
            vi,ti=(idx,other) if d=='V2T' else (other,idx)
            a,b=channels(v[vi],t[ti],vm[vi],tm[ti],x['logit_scale'])
            margins.append((pos-((a+b)/2).diag()).cpu())
        m=torch.cat(margins)
        res['directions'][d]={'n':len(m),'strict_hard_pair_accuracy':float((m>0).double().mean()*100),
                             'nonpositive_margin_n':int((m<=0).sum()),
                             'median_margin':float(m.median()),'mean_margin':float(m.mean()),
                             'margins':m.tolist()}
    res['interpretation']=[
        'Train errors remain, but most anchors already have positive margins.',
        'This is not proof of gradient starvation, feature insufficiency, or irreducibility.',
        'The 600-step readout screen failed; a direct hard-pair diagnostic can test training adequacy separately.',
    ]
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/AS-C01-HEALTH.json',res)
    print(json.dumps({d:{k:v for k,v in m.items() if k!='margins'} for d,m in res['directions'].items()},indent=2))


if __name__=='__main__':
    main()
