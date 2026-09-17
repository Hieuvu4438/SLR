"""Train-only hard-confuser diagnostic with paired, matched readout controls.

No claim of novelty. No dev examples enter training, batching, or hard mining.
Clean-text balanced CiCo objective is fixed for all diagnostic arms. This first
screen deliberately omits dynamic random-swap augmentation for every arm; it is
not a final baseline-equivalent training campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .common import ART, ROOT, dump, metrics, rows, sha
from .scoring import Readout, balanced_loss, channels

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


def load(split):
    c = torch.load(ART / f'frozen_{split}.pt', weights_only=True)
    assert c['manifest_sha256'] == sha(ROOT / f'artifacts/manifests/ph_{split}.jsonl')
    assert c['ids'] == [r['pair_id'] for r in rows(split)]
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in c.items()}


def args_for(c, i, j=None):
    j = i if j is None else j
    return c['video_tokens'][i], c['text_tokens'][j], c['video_mask'][i], c['text_mask'][j]


@torch.inference_mode()
def train_confusers(c):
    path = ART / 'train_confusers.pt'
    if path.exists():
        data = torch.load(path, weights_only=True)
        assert data['checkpoint_sha256'] == c['checkpoint_sha256']
        assert data['manifest_sha256'] == c['manifest_sha256']
        return data['video_confuser'], data['text_confuser']
    n = len(c['ids'])
    rv = torch.full((n,), -torch.inf, device='cuda')
    rt = rv.clone()
    iv = torch.zeros(n, dtype=torch.long, device='cuda')
    it = iv.clone()
    for i in range(0, n, 128):
        for j in range(0, n, 128):
            a, b = channels(*args_for(c, slice(i, i+128), slice(j, j+128)), c['logit_scale'])
            s = (a+b)/2
            if i == j:
                s.fill_diagonal_(-torch.inf)
            value, idx = s.max(1)
            better = value > rv[i:i+len(value)]
            iv[i:i+len(value)] = torch.where(better, idx+j, iv[i:i+len(value)])
            rv[i:i+len(value)] = torch.maximum(value, rv[i:i+len(value)])
            value, idx = s.max(0)
            better = value > rt[j:j+len(value)]
            it[j:j+len(value)] = torch.where(better, idx+i, it[j:j+len(value)])
            rt[j:j+len(value)] = torch.maximum(value, rt[j:j+len(value)])
        if i % 512 == 0:
            print(json.dumps({'event': 'train_only_confusers', 'done': min(i+128,n), 'n': n}), flush=True)
    data = {'video_confuser': iv.cpu(), 'text_confuser': it.cpu(),
            'checkpoint_sha256': c['checkpoint_sha256'], 'manifest_sha256': c['manifest_sha256'],
            'split': 'train', 'candidate_n': n, 'positive_rule': 'paired ID, diagonal excluded'}
    torch.save(data, path)
    return data['video_confuser'], data['text_confuser']


@torch.inference_mode()
def evaluate(model, c, base_channels):
    model.eval()
    output = np.empty_like(base_channels[0])
    for i in range(0, len(c['ids']), 64):
        for j in range(0, len(c['ids']), 64):
            a, b = model(*args_for(c, slice(i,i+64), slice(j,j+64)))
            correction = ((a+b)/2).cpu().numpy()
            output[i:i+64,j:j+64] = base_channels[:,i:i+64,j:j+64].mean(0) + correction
    return output


def run(mode, seed, train, dev, confusers, base_dev, steps, lr,
        experiment_prefix='AS-C01-R1', loss_baseline_weight=1.):
    torch.manual_seed(seed)
    np.random.seed(seed)
    expid = f'{experiment_prefix}-{mode}-s{seed}'
    run_dir = ART / expid
    if run_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {run_dir}')
    run_dir.mkdir()
    start = time.time()
    model = Readout(mode).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = np.random.default_rng(seed)
    report = {
        'experiment_id': expid, 'status': 'running', 'pid': os.getpid(), 'start_unix': start,
        'seed': seed, 'mode': mode, 'command': sys.argv,
        'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_hashes': {str(p.relative_to(ROOT)):sha(p) for p in
                        (ROOT/'methods/information_probe').glob('*.py')},
        'checkpoint_sha256': train['checkpoint_sha256'],
        'manifest_hashes': {s:c['manifest_sha256'] for s,c in [('train',train),('dev',dev)]},
        'feature_provenance': 'Exact contextual cache from AS-C01-R0; baseline encoders frozen',
        'optimizer': 'AdamW beta=(.9,.999) eps=1e-8 wd=.001', 'lr': lr,
        'schedule': 'cosine, no warmup', 'batch': 64, 'effective_batch': 64,
        'planned_updates': steps, 'precision': 'float32', 'gpu': torch.cuda.get_device_name(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'objective': 'mean CE(a), CE(a.T), CE(b), CE(b.T); paired positives',
        'loss_baseline_weight': loss_baseline_weight,
        'inference_baseline_weight': 1.,
        'augmentation': 'clean-only all arms; screening deviation from historical random swap',
        'batching': '32 train anchors plus alternating T2V/V2T train-only hardest negatives, fill unique to 64',
        'dev_selector': 'best official mean_R1, then mean_R5; initial checkpoint eligible; every 100 steps',
        'protocol': '519x519 complete dev scores, legacy ties, historical persistent slices',
        'history': [], 'initial': None, 'selected_step': None,
    }
    dump(OUT/f'{expid}_run.json',report)
    baseline = base_dev.mean(0)
    best = (-np.inf, -np.inf)
    best_scores = None
    try:
        for step in range(steps+1):
            if step % 100 == 0 or step == steps:
                scores = evaluate(model, dev, base_dev)
                m = metrics(scores, baseline)
                current = (m['official_mean_R1'], (m['official_T2V']['R5']+m['V2T']['R5'])/2)
                if step == 0:
                    assert np.array_equal(scores, baseline), 'nonzero initialization!'
                    report['initial'] = {'mean_R1': current[0], 'T2V_R1': m['official_T2V']['R1'], 'V2T_R1':m['V2T']['R1']}
                history = {'step': step, 'mean_R1': current[0], 'T2V_R1':m['official_T2V']['R1'],
                           'V2T_R1':m['V2T']['R1'], 'wall_seconds':time.time()-start}
                if step:
                    history['last_train_loss'] = float(loss.detach())
                report['history'].append(history)
                dump(OUT/f'{expid}_step{step}_metrics.json',m)
                np.save(run_dir/f'scores_step{step}.npy',scores)
                if current > best:
                    best = current
                    best_scores = scores.copy()
                    report['selected_step'] = step
                    torch.save(model.state_dict(),run_dir/'selected.pt')
                dump(OUT/f'{expid}_run.json',report)
                print(json.dumps({'event':'dev','experiment':expid,**history}),flush=True)
            if step == steps:
                break
            model.train()
            anchors = rng.choice(len(train['ids']),32,replace=False)
            hard = confusers[step % 2][anchors].numpy()
            selected = list(dict.fromkeys(np.concatenate((anchors,hard)).tolist()))
            while len(selected)<64:
                x = int(rng.integers(len(train['ids'])))
                if x not in selected:
                    selected.append(x)
            idx = torch.tensor(selected,device='cuda')
            inp = args_for(train,idx)
            with torch.no_grad():
                a,b = channels(*inp,train['logit_scale'])
            x,y = model(*inp)
            loss = balanced_loss(loss_baseline_weight*a+x,loss_baseline_weight*b+y)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            scheduler.step()
        report['status'] = 'completed'
        torch.save(model.state_dict(),run_dir/'last.pt')
        report['updates'] = steps
        report['selected_mean_R1'] = best[0]
        report['learned_gain_pp'] = best[0]-report['initial']['mean_R1']
        np.save(run_dir/'selected_scores.npy',best_scores)
        report['selected_checkpoint_sha256'] = sha(run_dir/'selected.pt')
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        torch.cuda.synchronize()
        report['wall_seconds'] = time.time()-start
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        dump(OUT/f'{expid}_run.json',report)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--modes',nargs='+',choices=['interaction','pooled','zero'],default=['pooled','interaction'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,1337,2026])
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--experiment-prefix',default='AS-C01-R1')
    p.add_argument('--loss-baseline-weight',type=float,choices=[0.,1.],default=1.)
    args=p.parse_args()
    torch.set_num_threads(8)
    train,dev=load('train'),load('dev')
    confusers=train_confusers(train)
    base_dev=np.load(ART/'baseline_dev_channels.npy')
    for seed in args.seeds:
        for mode in args.modes:
            run(mode,seed,train,dev,confusers,base_dev,args.steps,args.lr,
                args.experiment_prefix,args.loss_baseline_weight)


if __name__ == '__main__':
    main()
