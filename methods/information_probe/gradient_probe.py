"""Train-only directional sensitivities at video-token interface, not gradients of encoder parameters."""
from collections import Counter
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state,load_cico_tokenizer
from .common import ART,ROOT,dump,rows,sha
from .scoring import channels
from .text_augmentation_probe import captions,encode

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def gradient_summary(gv,gt,valid,duplicate):
    gv=(gv*valid[...,None]).flatten(1).double()
    gt=(gt*valid[...,None]).flatten(1).double()
    nv,nt=gv.norm(dim=1),gt.norm(dim=1)
    cosine=(gv*gt).sum(1)/(nv*nt).clamp_min(1e-30)
    active=(nv>1e-15)&(nt>1e-15)
    allcos=(gv*gt).sum()/(gv.norm()*gt.norm()).clamp_min(1e-30)
    both=(gv+gt)/2
    mass=both.square().sum(1)
    denominator=mass.sum().clamp_min(1e-30)
    return {'global_cosine':float(allcos),'negative_example_cosine_fraction':float((cosine[active]<0).double().mean()),
            'active_gradient_examples':int(active.sum()),'median_example_cosine':float(cosine[active].median()),
            'V2T_gradient_norm':float(gv.norm()),'T2V_gradient_norm':float(gt.norm()),
            'combined_gradient_norm':float(both.norm()),
            'duplicate_n':int(duplicate.sum()),'duplicate_example_fraction':float(duplicate.double().mean()),
            'duplicate_squared_norm_mass_fraction':float(mass[duplicate].sum()/denominator),
            'top10_examples_squared_norm_mass_fraction':float(mass.topk(min(10,len(mass))).values.sum()/denominator)}


def main():
    report_path=OUT/'AS-C12-GRADIENT_run.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    torch.set_num_threads(8)
    start=time.time()
    report={'experiment_id':'AS-C12-GRADIENT','status':'running','pid':os.getpid(),
            'code_sha256':sha(__file__),'scoring_sha256':sha(ROOT/'methods/information_probe/scoring.py'),
            'updates':0,'trainable_parameters':0,'gradient_target':'contextual video tokens only',
            'batch_size':512,'batches_per_seed':4,'seeds':[42,1337,2026],'conditions':['clean','deployed_aug'],
            'selector':None,'method_go':False,'batches':[]}
    dump(report_path,report)
    try:
        cache=torch.load(ART/'frozen_train.pt',weights_only=True)
        records=rows('train')
        assert cache['ids']==[r['pair_id'] for r in records]
        assert cache['manifest_sha256']==sha(ROOT/'artifacts/manifests/ph_train.jsonl')
        checkpoint=ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
        assert cache['checkpoint_sha256']==sha(checkpoint)
        report.update(checkpoint_sha256=cache['checkpoint_sha256'],manifest_sha256=cache['manifest_sha256'])
        config=yaml.safe_load((checkpoint.parents[1]/'resolved_config.yaml').read_text())
        cico_root=ROOT/config['upstream']['cico_root'];sys.path.insert(0,str(cico_root))
        raw=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
        state={k.removeprefix('core.'):v for k,v in raw['model'].items() if k.startswith('core.')}
        core=_load_cico_core_from_state(config,state,cico_root=cico_root,device='cuda')
        core.eval().requires_grad_(False)
        bridge,tokenizer=CiCoBridge(core),load_cico_tokenizer(config)
        clean=encode(bridge,tokenizer,captions(records,None))
        assert torch.equal(clean.tokens.cpu(),cache['text_tokens'])
        # encode() returns inference tensors; clone outside inference mode for use in autograd.
        clean_tokens,clean_mask=clean.tokens.clone(),clean.mask.clone()
        keys=[tuple(encode_cico_text(r['caption_model'],tokenizer,32)[0].tolist()) for r in records]
        counts=Counter(keys)
        for seed in report['seeds']:
            aug=encode(bridge,tokenizer,captions(records,seed))
            aug_tokens,aug_mask=aug.tokens.clone(),aug.mask.clone()
            order=np.random.default_rng(seed).permutation(len(records))
            for batch_idx in range(4):
                indexes=order[batch_idx*512:(batch_idx+1)*512]
                localcounts=Counter(keys[i] for i in indexes)
                duplicate=torch.tensor([localcounts[keys[i]]>1 for i in indexes],device='cuda')
                globalduplicate=torch.tensor([counts[keys[i]]>1 for i in indexes],device='cuda')
                vm=cache['video_mask'][indexes].cuda()
                text,tm=clean_tokens[indexes],clean_mask[indexes]
                for condition in report['conditions']:
                    v=cache['video_tokens'][indexes].cuda().requires_grad_(True)
                    a,b=channels(v,text,vm,tm,cache['logit_scale'])
                    if condition=='deployed_aug':
                        _,b=channels(v,aug_tokens[indexes],vm,aug_mask[indexes],cache['logit_scale'])
                    target=torch.arange(len(v),device='cuda')
                    lv=(F.cross_entropy(a,target)+F.cross_entropy(b,target))/2
                    lt=(F.cross_entropy(a.T,target)+F.cross_entropy(b.T,target))/2
                    gv,=torch.autograd.grad(lv,v,retain_graph=True)
                    gt,=torch.autograd.grad(lt,v)
                    summary=gradient_summary(gv,gt,vm==0,duplicate)
                    global_summary=gradient_summary(gv,gt,vm==0,globalduplicate)
                    entry={'seed':seed,'batch':batch_idx,'condition':condition,'indexes':indexes.tolist(),
                           'V2T_loss':float(lv.detach()),'T2V_loss':float(lt.detach()),
                           **summary,'global_train_duplicate_n':int(globalduplicate.sum()),
                           'global_train_duplicate_squared_norm_mass_fraction':global_summary['duplicate_squared_norm_mass_fraction']}
                    report['batches'].append(entry)
                    dump(report_path,report)
                    print(json.dumps({k:value for k,value in entry.items() if k!='indexes'}),flush=True)
                    del v,a,b,lv,lt,gv,gt
        report.update(status='completed',wall_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      limits=['Video-token gradients are not shared encoder parameter gradients.',
                              'Global duplicate membership does not imply a duplicate negative is present in this batch.',
                              'All batches reused fixed trained backbone; no independent model-training replication.',
                              'No correction, gradient surgery, or new method justified by sensitivity alone.'])
        dump(report_path,report)
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),wall_seconds=time.time()-start)
        dump(report_path,report)
        raise


if __name__=='__main__':
    main()
