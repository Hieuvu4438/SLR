"""Validate all AS-C05 runs and measure last-checkpoint train discrimination."""
import json

import numpy as np
import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.factory import load_cico_tokenizer
from .common import ART,ROOT,dump,metrics,ranks,rows,sha
from .scoring import Readout
from .summarize_readout import cluster_bootstrap
from .train_readout import args_for,load

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    records=rows('train')
    config=yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
    tok=load_cico_tokenizer(config)
    keys=[tuple(encode_cico_text(r['caption_model'],tok,32)[0].tolist()) for r in records]
    hard=torch.load(ART/'train_confusers.pt',weights_only=True)
    cache=load('train')
    baseline=np.load(ART/'baseline_dev_scores.npy')
    result={'experiment_id':'AS-C05-SUMMARY','status':'analyzed','code_sha256':sha(__file__),'runs':[],
            'manifest_sha256':cache['manifest_sha256'],'checkpoint_sha256':cache['checkpoint_sha256']}
    for coefficient in (0,1):
        for mode in ('pooled','interaction','zero'):
            for seed in (42,1337,2026):
                expid=f'AS-C05-loss{coefficient}-{mode}-s{seed}'
                report=json.loads((OUT/f'{expid}_run.json').read_text())
                assert report['status']=='completed' and report['updates']==2400
                selected=np.load(ART/expid/'selected_scores.npy')
                m=metrics(selected,baseline)
                model=Readout(mode).cuda()
                model.load_state_dict(torch.load(ART/expid/'last.pt',weights_only=True))
                model.eval()
                entry={'id':expid,'loss_baseline_coefficient':coefficient,'mode':mode,'seed':seed,
                       'selected_step':report['selected_step'],'selected_mean_R1':m['official_mean_R1'],
                       'learned_gain_pp':report['learned_gain_pp'],'final_dev_mean_R1':report['history'][-1]['mean_R1'],
                       'first_logged_train_loss':report['history'][1]['last_train_loss'],
                       'last_logged_train_loss':report['history'][-1]['last_train_loss'],
                       'loss_note':'different batches; trajectory available, not a paired loss estimate',
                       'last_train_pair_accuracy':{}}
                for d,key in [('T2V','text_confuser'),('V2T','video_confuser')]:
                    measured=[]
                    for i in range(0,len(records),64):
                        idx=torch.arange(i,min(i+64,len(records)),device='cuda')
                        other=hard[key][i:i+64].cuda()
                        a,b=model(*args_for(cache,idx))
                        pos=((a+b)/2).diag()
                        vi,ti=(other,idx) if d=='T2V' else (idx,other)
                        a,b=model(*args_for(cache,vi,ti))
                        measured.append((pos-((a+b)/2).diag()).cpu().numpy())
                    delta=np.concatenate(measured)
                    base_delta=np.load(ART/f'train_pair_channel_deltas_{d}.npy').mean(1)
                    distinct=np.array([keys[i]!=keys[int(j)] for i,j in enumerate(hard[key])])
                    entry['last_train_pair_accuracy'][d]={
                        'n':len(delta),'distinct_input_pairs':int(distinct.sum()),
                        'standalone_pct':float(100*np.mean(delta>1e-4)),
                        'standalone_distinct_input_pct':float(100*np.mean(delta[distinct]>1e-4)),
                        'combined_pct':float(100*np.mean(base_delta+delta>1e-4)),
                        'combined_distinct_input_pct':float(100*np.mean((base_delta+delta)[distinct]>1e-4)),
                        'baseline_distinct_input_pct':float(100*np.mean(base_delta[distinct]>1e-4)),
                    }
                    np.save(ART/expid/f'last_train_pair_delta_{d}.npy',delta)
                result['runs'].append(entry)
                print(json.dumps(entry),flush=True)
    result['all_select_initialization']=all(r['selected_step']==0 for r in result['runs'])
    result['coefficient0_vs1']={}
    labels=[r['video_id'].rsplit('-',1)[0] for r in rows('dev')]
    for mode in ('pooled','interaction','zero'):
        deltas=[];per_seed=[]
        for seed in (42,1337,2026):
            left=np.load(ART/f'AS-C05-loss0-{mode}-s{seed}'/'selected_scores.npy')
            right=np.load(ART/f'AS-C05-loss1-{mode}-s{seed}'/'selected_scores.npy')
            lr,rr=ranks(left),ranks(right)
            deltas.append(sum((lr[d]==0).astype(float)-(rr[d]==0) for d in ('T2V','V2T'))*50)
            per_seed.append({'seed':seed,'mean_R1_gain_pp':float(deltas[-1].mean()),
                             'half_point_gate':float(deltas[-1].mean())>=.5})
        result['coefficient0_vs1'][mode]={
            'bootstrap':cluster_bootstrap(np.mean(deltas,axis=0),labels),
            'per_seed':per_seed,
            'necessary_effect_gate_passed':sum(x['half_point_gate'] for x in per_seed)>=2}
    result['limits']=[
        'The coefficient0 intervention deliberately changes loss/inference composition; not a method claim.',
        'Train discrimination is on fixed R0 hard pairs, not full-gallery train retrieval or out-of-fold errors.',
        'Identical-input pair exclusions are descriptive only; official training positives were not changed.',
        'A negative result rejects this training-signal remedy, not all information recoverability.']
    result['fallacy_scan']=json.loads((OUT/'AS-C01-R1_summary.json').read_text())['fallacy_scan']
    result['fallacy_scan']['survivorship']='NOTE: all 18 registered 2400-update runs required'
    result['fallacy_scan']['correlation_causation']='CAUTION: direct coefficient intervention identifies this diagnostic training setup only'
    dump(OUT/'AS-C05-SUMMARY.json',result)


if __name__=='__main__':
    main()
