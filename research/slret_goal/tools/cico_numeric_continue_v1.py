"""Matched PH CiCo continuation using existing data/model/loss/evaluation code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from cico_fp32_moments import FP32MomentAdamW
from cico_moment_arithmetic import tensor_digest
from extraction_resume import atomic_json
from inventory import ROOT,sha
from optimizer_exposure import count_moments


def rng_digest(state):
    return hashlib.sha256(json.dumps(state,sort_keys=True,default=lambda x:x.tolist()).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--arm',choices=['native','fp32'],required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--reference')
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    start=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,arm=cli.arm,
        smoke=cli.smoke,test_loaded=False,dev_loaded=not cli.smoke,seed=42,batch_size=512,
        optimizer_updates=0,examples_seen=0,script_sha256=sha(__file__),evaluations={},
        protocol_sha256=sha(ROOT/'research/slret_goal/CICO_NUMERICAL_CONTINUATION_PROTOCOL.md'))
    def record(ledger=False):
        report['wall_seconds']=time.time()-start
        atomic_json(out/'run.json',report)
        if ledger:
            with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    record(True)
    try:
        cap=24 if cli.smoke else 30
        reserve=.016 if cli.smoke else 2.1
        assert shutil.disk_usage(out).free-reserve*1024**3>15*1024**3
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())+reserve*1024**3<cap*1024**3
        torch.set_num_threads(4)
        sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'methods/elsc'),str(ROOT/'third_party/SLRT/CiCo/CLCL')]
        from elsc.models.retriever import ELSCRetriever
        from elsc.train import _optimizer,_scheduler,_seed_everything,_amp_settings
        from elsc.losses.coarse import balanced_clcl_loss
        from elsc.evaluate import evaluate_model
        from slr_common.upstream.factory import _load_cico_core_from_state,load_cico_tokenizer
        from slr_common.data.cico_dataset import CiCoFeatureDataset
        from slr_common.data.tokenize import CiCoCollator
        from slr_common.utils import capture_rng_state,restore_rng_state
        root=ROOT/'runs/ph_base_b512_s42'
        cfg=yaml.safe_load((root/'resolved_config.yaml').read_text())
        assert cfg['method']=='baseline' and not cfg['model']['adapter']['enabled']
        assert cfg['train']['per_device_batch']==512 and cfg['train']['epochs']==200
        selected=json.loads((root/'selection.json').read_text())
        source=root/selected['checkpoint']
        assert sha(source)==selected['checkpoint_sha256']
        _seed_everything(42)
        raw=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        state={n.removeprefix('core.'):v for n,v in raw['model'].items() if n.startswith('core.')}
        device=torch.device('cuda')
        core=_load_cico_core_from_state(cfg,state,cico_root=ROOT/cfg['upstream']['cico_root'],device=device)
        model=ELSCRetriever(core,input_dim=1024,hidden_dim=256,text_dim=core.clip.text_projection.shape[-1],
                            core_frozen=False,adapter_enabled=False).to(device)
        model.load_state_dict(raw['model'],strict=True)
        native=_optimizer(model,cfg)
        assert [n for n,p in model.named_parameters() if p.requires_grad]==raw['trainable_parameter_names']
        for group in native.param_groups:
            group.update(foreach=False,fused=False)
        if cli.arm=='native':
            optimizer=native
        else:
            optimizer=FP32MomentAdamW(native.param_groups,lr=1e-5,betas=(.9,.98),eps=1e-6,weight_decay=.001)
        del native,raw,state
        report.update(initial_checkpoint_sha256=sha(source),config=cfg,config_sha256=sha(root/'resolved_config.yaml'),
                      initial_model_digest=tensor_digest(model.state_dict()),torch=torch.__version__,
                      hardware=torch.cuda.get_device_name(),kind='numerical_baseline_continuation_not_new_method')
        sources=['methods/elsc/elsc/train.py','methods/elsc/elsc/evaluate.py','methods/elsc/elsc/losses/coarse.py',
                 'shared/slr_common/upstream/cico_bridge.py','shared/slr_common/data/cico_dataset.py',
                 'shared/slr_common/data/tokenize.py','third_party/SLRT/CiCo/CLCL/modules/modeling.py',
                 'research/slret_goal/tools/cico_fp32_moments.py']
        report['source_sha256']={p:sha(ROOT/p) for p in sources}
        data=CiCoFeatureDataset(ROOT/cfg['data']['train_manifest'],feature_len=64,alpha=.9,split='train')
        collator=CiCoCollator(load_cico_tokenizer(cfg),32,augment=True,seed=42)
        generator=torch.Generator().manual_seed(42)
        loader=torch.utils.data.DataLoader(data,batch_size=512,shuffle=True,generator=generator,
            drop_last=True,num_workers=8,pin_memory=True,collate_fn=collator)
        assert len(loader)==13
        scheduler=_scheduler(optimizer,len(loader)*200,.1)
        report['schedule_total_steps']=2600
        report['manifests']={s:sha(ROOT/cfg['data'][f'{s}_manifest']) for s in ['train','dev']}
        if not cli.smoke:
            devdata=CiCoFeatureDataset(ROOT/cfg['data']['dev_manifest'],feature_len=64,alpha=.9,split='dev')
            assert not {r.video_id for r in data.records}&{r.video_id for r in devdata.records}
            files={str(p):sha(p) for ds in [data,devdata] for r in ds.records
                   for p in [r.feature_aware,r.feature_agnostic]}
            atomic_json(out/'feature_hashes.json',files)
            report['feature_hashes_sha256']=sha(out/'feature_hashes.json')
            del devdata,files
        reference=None
        if cli.reference:
            refroot=ROOT/'artifacts/slret_goal'/cli.reference
            reference=json.loads((refroot/'run.json').read_text())
            assert reference['status']=='completed' and reference['smoke']==cli.smoke
            for key in ['initial_model_digest','config','source_sha256','manifests']:
                assert report[key]==reference[key],('Paired provenance',key)
            if not cli.smoke:assert report['feature_hashes_sha256']==reference['feature_hashes_sha256']
            refrows=[json.loads(x) for x in (refroot/'train_steps.jsonl').read_text().splitlines()]
        selection=None
        def evaluate(step):
            nonlocal selection
            saved=capture_rng_state()
            scores,metrics=evaluate_model(model,cfg,'dev',device)
            restore_rng_state(saved)
            if step==0:
                assert np.array_equal(scores,np.load(root/'evaluation/dev/scores_video_x_text.npy'))
            folder=out/f'eval_step{step:03d}'
            folder.mkdir()
            np.save(folder/'scores_video_x_text.npy',scores)
            compact={k:v for k,v in metrics.items() if k not in ['per_query','diagnostic_best_positive']}
            atomic_json(folder/'metrics.json',compact)
            metric={d:{k:v for k,v in metrics[d].items() if k!='cols'} for d in ['T2V','V2T']}
            report['evaluations'][str(step)]=metric
            r1={d:metric[d]['R1'] for d in metric}
            mean=sum(r1.values())/2
            initial=report['evaluations']['0']
            eligible=all(r1[d]>=initial[d]['R1']-.5 for d in r1)
            if selection is None or (eligible and mean>selection['mean_R1']):
                selection=dict(step=step,R1=r1,mean_R1=mean)
                if step==0:
                    selection.update(checkpoint=str(source),sha256=sha(source),inference_only=False)
                else:
                    tmp=out/'selected.tmp.pt'
                    torch.save(dict(model=model.state_dict(),step=step,config=cfg,initial_sha256=sha(source)),tmp)
                    tmp.replace(out/'selected.pt')
                    selection.update(checkpoint=str(out/'selected.pt'),sha256=sha(out/'selected.pt'),inference_only=True)
                atomic_json(out/'selection.json',selection)
            print(json.dumps(dict(event='DEV',step=step,mean_R1=mean,R1=r1,eligible=eligible)),flush=True)
            record()
        if not cli.smoke:evaluate(0)
        torch.cuda.reset_peak_memory_stats()
        enabled,dtype=_amp_settings(cfg,device)
        assert enabled and dtype==torch.bfloat16
        step=0
        for epoch in range(1 if cli.smoke else 20):
            collator.set_epoch(epoch)
            for batch in loader:
                started=time.time()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                inputs={'h':batch['h'],'valid':batch['valid']}
                inputs.update({f'{name}_{i}':v for name in ['clean_text','aug_text'] for i,v in enumerate(batch[name])})
                inputhash=tensor_digest(inputs)
                with torch.autocast('cuda',dtype=dtype):
                    video,_=model.encode_video(batch['h'].to(device),batch['valid'].to(device))
                    clean=model.encode_text(*(v.to(device) for v in batch['clean_text']))
                    aug=model.encode_text(*(v.to(device) for v in batch['aug_text']))
                    a,b=model.bridge.score(video,clean,text_aug=aug,objective=True)
                    loss=balanced_clcl_loss(a,b,dual_mix=.5,mix_design='balance')
                assert torch.isfinite(loss)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                assert torch.isfinite(norm)
                row=dict(step=step+1,epoch=epoch,ids=batch['pair_id'],input_sha256=inputhash,
                         loss=float(loss),gradient_norm=float(norm),lr=[g['lr'] for g in optimizer.param_groups])
                if step==0:
                    row['gradient_sha256']=tensor_digest({n:p.grad for n,p in model.named_parameters() if p.grad is not None})
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step+=1
                row['rng_digest']=rng_digest(capture_rng_state())
                if reference:
                    target=refrows[step-1]
                    for key in ['step','epoch','ids','input_sha256','lr','rng_digest']:
                        assert row[key]==target[key],('Paired exposure/RNG',step,key)
                    if step==1:
                        for key in ['loss','gradient_norm','gradient_sha256']:
                            assert row[key]==target[key],('Initial gradient parity',key)
                torch.cuda.synchronize()
                row['seconds']=time.time()-started
                with (out/'train_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                report.update(optimizer_updates=step,examples_seen=step*512)
                record()
                if step%13==0 or cli.smoke:
                    print(json.dumps({k:v for k,v in row.items() if k!='ids'}),flush=True)
                del video,clean,aug,a,b,loss
                if cli.smoke and step==2:break
            if not cli.smoke:
                evaluate(step)
        assert step==(2 if cli.smoke else 260)
        counts=count_moments(optimizer.state)
        assert all(v['nonfinite_m']==v['nonfinite_v']==0 for v in counts.values())
        if cli.arm=='fp32':assert set(counts)=={'torch.float32'}
        report.update(moment_counts=counts,endpoint_model_digest=tensor_digest(model.state_dict()),
                      endpoint_rng_digest=rng_digest(capture_rng_state()),peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if not cli.smoke:
            checkpoint=out/'final.pt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                            rng=capture_rng_state(),sampler_generator_state=generator.get_state(),completed_steps=step,
                            completed_epochs=20,config=cfg,arm=cli.arm,initial_checkpoint_sha256=sha(source)),checkpoint)
            report.update(final_checkpoint=str(checkpoint),final_checkpoint_sha256=sha(checkpoint),selection=selection)
        report.update(status='completed',exit_status=0,
                      decision='Matched numerical baseline only; causal recall comparison requires paired audit.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record(True)
        print(json.dumps({k:report.get(k) for k in ['run_id','status','optimizer_updates','wall_seconds','error','selection']}),flush=True)


if __name__=='__main__':main()
