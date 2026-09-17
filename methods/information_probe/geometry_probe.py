"""Unsupervised train-only geometry controls, not a novel retrieval method."""
import json
import os
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .common import ART,ROOT,dump,metrics,ranks,sha
from .scoring import channels

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def moments(tokens, valid):
    """Each sequence has mass1/N, each of its valid slots equal conditional mass."""
    if tokens.ndim!=3 or valid.shape!=tokens.shape[:2] or not valid.any(1).all():
        raise ValueError('nonempty aligned sequence masks required')
    d=tokens.shape[-1]
    mean=torch.zeros(d,dtype=torch.float64)
    second=torch.zeros(d,d,dtype=torch.float64)
    for i in range(0,len(tokens),128):
        x=F.normalize(tokens[i:i+128].double(),dim=-1)
        weight=valid[i:i+128].double()/valid[i:i+128].sum(1)[:,None]/len(tokens)
        x=x.reshape(-1,d);weight=weight.reshape(-1)
        mean+=(x*weight[:,None]).sum(0)
        second+=x.T@(x*weight[:,None])
    return mean, second


def fitted_transforms(vm,vs,tm,ts,seed=42):
    mean=(vm+tm)/2
    covariance=(vs+ts)/2-mean[:,None]*mean[None,:]
    covariance=(covariance+covariance.T)/2
    values,vectors=torch.linalg.eigh(covariance)
    assert values.min()>-1e-9
    values=values.clamp_min(0)
    ridge=.1*values.mean()
    scales=(.9*values+ridge).rsqrt()
    identity=torch.eye(len(mean),dtype=mean.dtype)
    generator=torch.Generator().manual_seed(seed)
    random_basis=torch.linalg.qr(torch.randn(len(mean),len(mean),dtype=mean.dtype,generator=generator)).Q
    pc=vectors[:,-1];random_pc=random_basis[:,-1]
    transforms={'identity':(torch.zeros_like(mean),torch.zeros_like(mean),identity),
                'shared_center':(mean,mean,identity),
                'separate_center':(vm,tm,identity),
                'shared_pc1':(mean,mean,identity-pc[:,None]*pc[None,:]),
                'random_pc1':(mean,mean,identity-random_pc[:,None]*random_pc[None,:]),
                'shared_whiten':(mean,mean,(vectors*scales)@vectors.T),
                'random_whiten':(mean,mean,(random_basis*scales)@random_basis.T)}
    return covariance,values,transforms


def transform(tokens,mean,matrix):
    return (F.normalize(tokens,dim=-1)-mean.to(tokens))@matrix.to(tokens)


@torch.inference_mode()
def main():
    path=OUT/'AS-C11-GEOMETRY_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    start=time.time()
    result={'experiment_id':'AS-C11-GEOMETRY','status':'running','pid':os.getpid(),
            'code_sha256':sha(__file__),'scorer_sha256':sha(ROOT/'methods/information_probe/scoring.py'),
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'optimizer':None,'updates':0,'trainable_encoder_parameters':0,'selector':None,
            'control_seed':42,'fit_precision':'CPU float64','score_precision':'CUDA float32',
            'gpu':torch.cuda.get_device_name(),'variants':{},'method_go':False}
    dump(path,result)
    try:
        train=torch.load(ART/'frozen_train.pt',weights_only=True)
        dev=torch.load(ART/'frozen_dev.pt',weights_only=True)
        assert train['checkpoint_sha256']==dev['checkpoint_sha256']
        for split,cache in [('train',train),('dev',dev)]:
            assert cache['manifest_sha256']==sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl')
        result['provenance']={split:{'cache_sha256':sha(ART/f'frozen_{split}.pt'),
                                     'manifest_sha256':cache['manifest_sha256']}
                              for split,cache in [('train',train),('dev',dev)]}
        result['checkpoint_sha256']=train['checkpoint_sha256']
        vm,vs=moments(train['video_tokens'],train['video_mask']==0)
        tm,ts=moments(train['text_tokens'],train['text_mask']==1)
        covariance,eigenvalues,transforms=fitted_transforms(vm,vs,tm,ts)
        proportions=eigenvalues/eigenvalues.sum()
        result['train_geometry']={'n':len(train['ids']),'video_mean_norm':float(vm.norm()),
            'text_mean_norm':float(tm.norm()),'modality_mean_cosine':float(F.cosine_similarity(vm[None],tm[None])),
            'shared_mean_norm':float(((vm+tm)/2).norm()),
            'covariance_participation_ratio':float(eigenvalues.sum().square()/eigenvalues.square().sum()),
            'covariance_entropy_effective_rank':float(torch.exp(-(proportions*proportions.clamp_min(1e-30).log()).sum())),
            'top1_variance_fraction':float(proportions[-1]),'top10_variance_fraction':float(proportions[-10:].sum())}
        torch.save({'train_manifest_sha256':train['manifest_sha256'],'checkpoint_sha256':train['checkpoint_sha256'],
                    'video_mean':vm,'text_mean':tm,'covariance':covariance,'eigenvalues':eigenvalues,
                    'transforms':transforms},ART/'AS-C11-transforms.pt')
        del train
        v,t=dev['video_tokens'].cuda(),dev['text_tokens'].cuda()
        vmask,tmask=dev['video_mask'].cuda(),dev['text_mask'].cuda()
        base=np.load(ART/'baseline_dev_scores.npy')
        base_ch=np.load(ART/'baseline_dev_channels.npy')
        for name,(vmean,tmean,matrix) in transforms.items():
            vv,tt=(v,t) if name=='identity' else (transform(v,vmean,matrix),transform(t,tmean,matrix))
            scored=np.empty_like(base_ch)
            for i in range(0,len(v),64):
                for j in range(0,len(t),64):
                    a,b=channels(vv[i:i+64],tt[j:j+64],vmask[i:i+64],tmask[j:j+64],dev['logit_scale'])
                    scored[:,i:i+64,j:j+64]=torch.stack((a,b)).cpu().numpy()
            scores=scored.mean(0)
            if name=='identity':
                result['identity_max_channel_delta']=float(np.abs(scored-base_ch).max())
                assert result['identity_max_channel_delta']<=2e-5
                assert all(np.array_equal(ranks(scores)[d],ranks(base)[d]) for d in ('T2V','V2T'))
            m=metrics(scores,base)
            np.save(ART/f'AS-C11-{name}_scores.npy',scores)
            dump(OUT/f'AS-C11-{name}_metrics.json',m)
            result['variants'][name]={'T2V_R1':m['official_T2V']['R1'],'V2T_R1':m['V2T']['R1'],
                                      'mean_R1':m['official_mean_R1'],
                                      'persistent_mean_rank_delta':{d:m[d]['persistent_mean_rank_delta'] for d in ('T2V','V2T')}}
            dump(path,result)
            print(json.dumps({'variant':name,**result['variants'][name]}),flush=True)
        result.update(status='completed',wall_seconds=time.time()-start,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      limits=['No novel mechanism claim for standard geometric controls.',
                              'One fixed backbone, not three independent training seeds.',
                              'All variants disclosed; no dev selector or confirmatory significance claim.',
                              'Transforming padding can affect inner-softmax interactions.',
                              'Negative ranking effects do not prove absence of discriminative information.'])
        dump(path,result)
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc(),wall_seconds=time.time()-start)
        dump(path,result)
        raise


if __name__=='__main__':
    main()
