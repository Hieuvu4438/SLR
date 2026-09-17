"""Crossed raw screen. Refuses incomplete/misaligned or unverified caches.

Register this configuration BEFORE outcomes. This is a diagnostic continuation
of the clean-text screen, not a full augmentation-matched method campaign.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .common import ART,ROOT,dump,metrics,sha
from .raw_readout import RawReadout,raw_input
from .scoring import balanced_loss,channels
from .train_readout import args_for,load,train_confusers

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def load_raw(split,baseline):
    r=torch.load(ART/f'raw_{split}.pt',weights_only=True)
    assert r['complete'] and r['ids']==baseline['ids']
    assert r['manifest_sha256']==baseline['manifest_sha256']
    assert r['checkpoint_sha256']=='6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f'
    assert all(x['pooled_relative_delta']<=1e-4 for x in r['provenance'])
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in r.items()}


@torch.inference_mode()
def evaluate(model,arm,cache,raw,base):
    model.eval()
    scores=np.empty_like(base)
    for i in range(0,len(cache['ids']),64):
        vi=torch.arange(i,min(i+64,len(cache['ids'])),device='cuda')
        # Train-only shuffle: dev evidence remains paired, testing whether
        # learning from invalid correspondence transfers. Never permute labels.
        r=raw_input(cache,raw,arm,vi)
        for j in range(0,len(cache['ids']),64):
            ti=torch.arange(j,min(j+64,len(cache['ids'])),device='cuda')
            a,b=model(r,*args_for(cache,vi,ti))
            scores[i:i+64,j:j+64]=base[i:i+64,j:j+64]+((a+b)/2).cpu().numpy()
    return scores


def run(regime,arm,seed,train,dev,raw_train,raw_dev,confusers,base,steps,lr):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=RawReadout(regime,score_scale=train['logit_scale']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
    # Separate generator avoids changing batches between shuffled and real arms.
    permutation=torch.tensor(np.random.default_rng(seed+9000).permutation(len(train['ids'])),device='cuda')
    expid=f'AS-C02-{regime}-{arm}-s{seed}'
    directory=ART/expid
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir()
    start=time.time()
    report={
        'experiment_id':expid,'status':'running','start_unix':start,'pid':os.getpid(),
        'command':sys.argv,'seed':seed,'regime':regime,'arm':arm,
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'methods/information_probe').glob('*.py')},
        'checkpoint_sha256':train['checkpoint_sha256'],
        'raw_checkpoint_sha256':raw_train['checkpoint_sha256'],
        'manifests':{s:c['manifest_sha256'] for s,c in [('train',train),('dev',dev)]},
        'feature_provenance':'AS-C01-R0 exact frozen tokens + AS-C02 full-historical-batch same-I3D spatial/pooled cache',
        'parameters':sum(p.numel() for p in model.parameters()),
        'R2_fixed_score_scale':train['logit_scale'],
        'shuffle_permutation_sha256':__import__('hashlib').sha256(permutation.cpu().numpy().tobytes()).hexdigest(),
        'optimizer':'AdamW betas .9/.999 eps 1e-8 wd .001','lr':lr,'schedule':'cosine, no warmup',
        'batch':64,'effective_batch':64,'precision':'float32','gpu':torch.cuda.get_device_name(),
        'planned_updates':steps,'objective':'balanced four-direction CE, identical to clean-text R0 objective',
        'augmentation':'clean-only screen, same for every arm; full historical augmentation not claimed',
        'selector':'best mean official R1 then mean R5; every 100 updates; initialization eligible',
        'protocol':'full 519x519 gallery, paired positives, historical directional ties',
        'control_note':'arm architectures and parameter counts identical within regime; extraction costs reported separately',
        'history':[],
    }
    dump(OUT/f'{expid}_run.json',report)
    best=(-float('inf'),-float('inf'))
    try:
        for step in range(steps+1):
            if step%100==0 or step==steps:
                scores=evaluate(model,arm,dev,raw_dev,base)
                m=metrics(scores,base)
                value=(m['official_mean_R1'],(m['official_T2V']['R5']+m['V2T']['R5'])/2)
                if step==0:
                    assert np.array_equal(scores,base)
                    report['initial_mean_R1']=value[0]
                entry={'step':step,'mean_R1':value[0],'T2V_R1':m['official_T2V']['R1'],'V2T_R1':m['V2T']['R1'],
                       'wall_seconds':time.time()-start}
                if step:
                    entry['last_train_loss']=float(loss.detach())
                report['history'].append(entry)
                dump(OUT/f'{expid}_step{step}_metrics.json',m)
                np.save(directory/f'scores_step{step}.npy',scores)
                if value>best:
                    best=value
                    report['selected_step']=step
                    torch.save(model.state_dict(),directory/'selected.pt')
                    np.save(directory/'selected_scores.npy',scores)
                dump(OUT/f'{expid}_run.json',report)
                print(json.dumps({'event':'raw_dev','experiment':expid,**entry}),flush=True)
            if step==steps:
                break
            anchors=rng.choice(len(train['ids']),32,replace=False)
            hard=confusers[step%2][anchors].numpy()
            chosen=list(dict.fromkeys(np.concatenate((anchors,hard)).tolist()))
            while len(chosen)<64:
                candidate=int(rng.integers(len(train['ids'])))
                if candidate not in chosen:
                    chosen.append(candidate)
            idx=torch.tensor(chosen,device='cuda')
            inp=args_for(train,idx)
            raw=raw_input(train,raw_train,arm,idx,permutation=permutation if arm=='shuffled' else None)
            with torch.no_grad():
                a,b=channels(*inp,train['logit_scale'])
            model.train()
            x,y=model(raw,*inp)
            loss=balanced_loss(a+x,b+y)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();schedule.step()
        torch.save(model.state_dict(),directory/'last.pt')
        report.update(status='completed',updates=steps,selected_mean_R1=best[0],
                      learned_gain_pp=best[0]-report['initial_mean_R1'],
                      selected_checkpoint_sha256=sha(directory/'selected.pt'))
    except Exception:
        report['status']='failed';report['traceback']=traceback.format_exc()
        raise
    finally:
        torch.cuda.synchronize()
        report['wall_seconds']=time.time()-start
        report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
        dump(OUT/f'{expid}_run.json',report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--regimes',nargs='+',choices=['R2','R3'],default=['R2','R3'])
    p.add_argument('--arms',nargs='+',choices=['existing','pooled','spatial','shuffled','zero'],
                   default=['existing','pooled','spatial','shuffled','zero'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,1337,2026])
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--lr',type=float,default=.0003)
    a=p.parse_args()
    torch.set_num_threads(8)
    train,dev=load('train'),load('dev')
    rt,rd=load_raw('train',train),load_raw('dev',dev)
    cf=train_confusers(train)
    base=np.load(ART/'baseline_dev_scores.npy')
    for seed in a.seeds:
        for regime in a.regimes:
            for arm in a.arms:
                run(regime,arm,seed,train,dev,rt,rd,cf,base,a.steps,a.lr)


if __name__=='__main__':
    main()
