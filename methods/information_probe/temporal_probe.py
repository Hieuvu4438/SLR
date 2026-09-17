"""Frozen input-order stress test, not a semantically valid new benchmark."""
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.upstream.cico_bridge import CiCoBridge,TextEncoding,VideoEncoding
from slr_common.upstream.factory import _load_cico_core_from_state,load_cico_tokenizer
from .common import ART,ROOT,dump,metrics,sha

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def perturb(h,valid,ids,mode):
    output=h.clone();orders=[]
    for i,identifier in enumerate(ids):
        n=int(valid[i].sum())
        order=torch.arange(n,device=h.device)
        if mode=='reverse_input':
            order=order.flip(0)
        elif mode=='shuffle_input':
            seed=int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8],'little')
            order=torch.tensor(np.random.default_rng(seed).permutation(n),device=h.device)
        elif mode not in ('identity','mean_input'):
            raise ValueError(mode)
        output[i,:n]=h[i,order] if mode!='mean_input' else h[i,:n].mean(0)
        orders.append(order)
    return output,orders


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    start=time.time()
    checkpoint=ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
    config=yaml.safe_load((checkpoint.parents[1]/'resolved_config.yaml').read_text())
    cico_root=ROOT/config['upstream']['cico_root']
    sys.path.insert(0,str(cico_root))
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
    state={k.removeprefix('core.'):v for k,v in saved['model'].items() if k.startswith('core.')}
    core=_load_cico_core_from_state(config,state,cico_root=cico_root,device='cuda')
    core.eval().requires_grad_(False);bridge=CiCoBridge(core)
    cache=torch.load(ART/'frozen_dev.pt',weights_only=True)
    baseline=np.load(ART/'baseline_dev_scores.npy')
    assert sha(checkpoint)==cache['checkpoint_sha256']
    dataset=CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_dev.jsonl',feature_len=64,alpha=.9,split='dev')
    loader=DataLoader(dataset,batch_size=128,num_workers=4,shuffle=False,
                      collate_fn=CiCoCollator(load_cico_tokenizer(config),32,augment=False))
    result={'experiment_id':'AS-C06-TEMPORAL','status':'running',
            'code_sha256':sha(__file__),'checkpoint_sha256':cache['checkpoint_sha256'],
            'manifest_sha256':cache['manifest_sha256'],
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'trainable_parameters':0,'updates':0,'selector':None,
            'registered_variants':['identity','reverse_input','shuffle_input','mean_input'],
            'hypothesis':'R0 may ignore global order beyond within-window motion; stress test only',
            'variants':{}}
    dump(OUT/'AS-C06-TEMPORAL_run.json',result)
    for mode in result['registered_variants']:
        enc=[];offset=0;slot_deltas=[];aligned_deltas=[]
        for batch in loader:
            h,valid=batch['h'].cuda(),batch['valid'].cuda()
            altered,orders=perturb(h,valid,batch['pair_id'],mode)
            v=bridge.encode_video(altered,valid)
            original=cache['video_tokens'][offset:offset+len(h)].cuda()
            for i,order in enumerate(orders):
                n=len(order)
                before=original[i,1:n+1]
                after=v.tokens[i,1:n+1]
                slot_deltas.append(float((1-F.cosine_similarity(before,after,dim=-1)).mean()))
                aligned=after[torch.argsort(order)]
                aligned_deltas.append(float((1-F.cosine_similarity(before,aligned,dim=-1)).mean()))
            enc.append(VideoEncoding(v.mask.cpu(),v.tokens.cpu(),v.cls.cpu()))
            offset+=len(h)
        v=VideoEncoding(*(torch.cat([getattr(e,k) for e in enc]).cuda() for k in ('mask','tokens','cls')))
        t=TextEncoding(*(cache['text_'+k].cuda() for k in ('mask','tokens','cls')))
        scores=np.empty_like(baseline)
        for i in range(0,519,128):
            for j in range(0,519,128):
                vb=VideoEncoding(v.mask[i:i+128],v.tokens[i:i+128],v.cls[i:i+128])
                tb=TextEncoding(t.mask[j:j+128],t.tokens[j:j+128],t.cls[j:j+128])
                a,b=bridge.score(vb,tb)
                scores[i:i+128,j:j+128]=((a+b)/2).cpu().numpy()
        m=metrics(scores,baseline)
        np.save(ART/f'AS-C06-{mode}_scores.npy',scores)
        dump(OUT/f'AS-C06-{mode}_metrics.json',m)
        entry={'T2V_R1':m['official_T2V']['R1'],'V2T_R1':m['V2T']['R1'],
               'mean_R1':m['official_mean_R1'],'max_score_delta':float(np.abs(scores-baseline).max()),
               'changed_ranks':{d:int(np.count_nonzero(m[d]['rank_delta'])) for d in ('T2V','V2T')},
               'mean_same_slot_cosine_distance':float(np.mean(slot_deltas)),
               'mean_input_aligned_cosine_distance':float(np.mean(aligned_deltas))}
        if mode=='identity':
            assert entry['max_score_delta']==0 and not any(entry['changed_ranks'].values())
        result['variants'][mode]=entry
        dump(OUT/'AS-C06-TEMPORAL_run.json',result)
        print(json.dumps({'mode':mode,**entry}),flush=True)
    result.update(status='completed',wall_seconds=time.time()-start,
                  limits=['Reordering encoded I3D windows preserves within-window motion, not global chronology.',
                          'Perturbed clips are not validated sign-language minimal pairs; no linguistic causal claim.',
                          'A sensitivity result is not evidence that using more order improves retrieval.',
                          'This distinguishes pre-context input sensitivity from post-context scorer invariance.'])
    dump(OUT/'AS-C06-TEMPORAL_run.json',result)


if __name__=='__main__':
    main()
