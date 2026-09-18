"""Preregistered 222-step same-I3D fine-tuning baseline, locked TEST unused."""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from inventory import ROOT,sha
from extraction_resume import atomic_json,digest
from seds_runtime import patch_pickle,native_kwargs,model_inputs,verify_assets,eval_dataset,evaluate
from seds_optimizer_precision import ensure_fp32_moments
from seds_continue import rng_state,restore_rng


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,optimizer_updates=0,
                examples_seen=0,test_loaded=False,selection_split='historically_exposed_PH_dev',
                kind='B_tuned_same_I3D_tail_not_novel_method',seed=42,batch_size=32,
                script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'research/slret_goal/RGB_FINETUNE_PILOT_PROTOCOL.md'),
                source_hashes={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in
                               ['rgb_bridge_worker.py','rgb_bridge_state.py','rgb_gradient_replay.py','seds_runtime.py','seds_optimizer_precision.py']},
                evaluations={},paired_checkpoints={})
    worker=None

    def record(ledger=False):
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
        if ledger:
            with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')

    record(True)
    try:
        allowance=int(8.6*1024**3)
        assert shutil.disk_usage(out).free-allowance>15*1024**3
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())+allowance<24*1024**3
        torch.set_num_threads(4)
        sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'third_party/SEDS')]
        os.chdir(ROOT/'third_party/SEDS')
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose,ph_train_pose_collate_fn
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose,ph_pose_collate_fn
        patch_pickle(ph_DataLoader_train_pose,ph_DataLoader_pose)
        control=ROOT/'artifacts/slret_goal/seds-moment-control-001'
        ctrl=json.loads((control/'run.json').read_text())
        args=argparse.Namespace(**ctrl['config'])
        args.output_dir,args.local_rank=str(out),int(os.environ.get('LOCAL_RANK',0))
        args=native.set_seed_logger(args)
        device,_=native.init_device(args,args.local_rank)
        root=ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        devroot=ROOT/'artifacts/slret_goal/seds-adapted-dev-001'
        labels=pickle.load((root/'labels/train.pkl').open('rb'))
        batches=json.loads((control/'batch_indices.json').read_text())
        ids=list(labels)
        assert len(batches)==222 and sorted(i for b in batches for i in b)==list(range(7096))
        trainreport,assets=verify_assets(root,'train',ids)
        data=ph_DataLoader_train_pose(**native_kwargs(args,native.ClipTokenizer(),root,'train'))
        dev,devids=eval_dataset(ph_DataLoader_pose,args,native.ClipTokenizer(),devroot)
        devreport,devassets=verify_assets(devroot,'dev',devids)
        assert not set(ids)&set(devids)
        assert digest(assets)==ctrl['train_assets_digest'] and digest(devassets)==ctrl['dev_assets_digest']
        for k in ['pose_checkpoint_sha256','pose_config_sha256','rgb_checkpoint_sha256','rgb_source_sha256','rgb_recipe_sha256','native_loader_sha256']:
            assert trainreport[k]==devreport[k]
        atomic_json(out/'asset_hashes.json',dict(train=assets,dev=devassets))
        atomic_json(out/'batch_indices.json',batches)
        model=native.init_model(args,device)
        expected=torch.load(args.init_model,map_location='cpu')
        assert set(model.state_dict())==set(expected)
        assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in expected.items())
        del expected
        report.update(config=vars(args),hardware=torch.cuda.get_device_name(),torch=torch.__version__,
                      control_sha256=sha(control/'run.json'),batch_order_sha256=sha(out/'batch_indices.json'),
                      checkpoint_sha256=sha(args.init_model),train_assets_digest=digest(assets),dev_assets_digest=digest(devassets),
                      commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest())
        base=ROOT/'third_party/SEDS'
        for name in ['main_task_retrieval.py','modules/modeling.py','modules/optimization.py','dataloaders/dataloader_ph_retrieval_train_pose.py']:
            report['source_hashes'][str(base/name)]=sha(base/name)
        loader=torch.utils.data.DataLoader(dev,batch_size=32,shuffle=False,num_workers=0,collate_fn=ph_pose_collate_fn)
        saved=rng_state()
        initial=evaluate(native,args,model,loader,device,devids,out/'eval_step000')
        restore_rng(saved)
        for stream in ['fusion','pose','rgb']:
            current=np.load(out/'eval_step000'/f'{stream}_video_x_text.npy')
            target=np.load(control/'eval_step000'/f'{stream}_video_x_text.npy')
            assert np.array_equal(current,target),('Step0 score parity',stream)
        report['evaluations']['0']=initial
        initial_r1={d:initial['fusion'][d]['R1'] for d in ['T2V','V2T']}
        selection=dict(step=0,mean_R1=sum(initial_r1.values())/2,R1=initial_r1,
                       head_checkpoint=args.init_model,head_sha256=sha(args.init_model),
                       rgb_backbone=str(ROOT/'artifacts/pretrained/bsl5k.pth.tar'))
        atomic_json(out/'selection.json',selection)
        optimizer,_,wrapped=native.prep_optimizer(args,model,222,device,1,args.local_rank,coef_lr=args.coef_lr)
        with (out/'worker_stderr.log').open('x') as log:
            worker=subprocess.Popen(['/home/haipd/miniconda3/bin/python',str(Path(__file__).with_name('rgb_bridge_worker.py')),str(out),'--pilot'],
                                    cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        def receive():
            line=worker.stdout.readline()
            if not line: raise RuntimeError(f'Worker closed pipe, returncode={worker.poll()}')
            response=json.loads(line)
            if response['status']=='failed': raise RuntimeError(response['error'])
            return response
        def request(op,version,**kwargs):
            worker.stdin.write(json.dumps(dict(op=op,version=version,**kwargs))+'\n')
            worker.stdin.flush()
            return receive()
        report['worker']=receive()
        reference=[json.loads(line) for line in (control/'train_steps.jsonl').read_text().splitlines()]
        torch.cuda.reset_peak_memory_stats()
        record()
        for version,indices in enumerate(batches):
            assert shutil.disk_usage(out).free>15*1024**3
            stepstart=time.time()
            wrapped.train()
            batch=ph_train_pose_collate_fn([data[i] for i in indices])
            batchids=[ids[i] for i in indices]
            assert batchids==reference[version]['ids']
            encoding=request('encode',version,ids=batchids)
            assert sha(encoding['path'])==encoding['sha256']
            payload=torch.load(encoding['path'],map_location='cpu')
            assert payload['version']==version and payload['ids']==batchids
            rgb=payload['features'].to(device).detach().requires_grad_(True)
            batch['RGB_feature']=rgb
            optimizer.zero_grad()
            losses=wrapped(*model_inputs(batch,device))
            assert all(torch.isfinite(torch.as_tensor(x)) for x in losses)
            loss_delta=max(abs(float(a)-b) for a,b in zip(losses,reference[version]['loss']))
            if version<2: assert loss_delta<=1e-7,('Matched loss',version,loss_delta)
            losses[0].backward()
            assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
            path=out/f'gradient_v{version}.pt'
            torch.save(dict(version=version,gradient=rgb.grad.detach().cpu()),path)
            backward=request('backward',version,sha256=sha(path))
            norm=torch.nn.utils.clip_grad_norm_(wrapped.parameters(),1.)
            assert torch.isfinite(norm)
            if version<2: assert abs(float(norm)-reference[version]['gradient_norm'])<=1e-7
            ensure_fp32_moments(optimizer)
            optimizer.step()
            optimizer.zero_grad()
            torch.clamp_(model.clip.logit_scale.data,max=np.log(100))
            commit=request('commit',version)
            torch.cuda.synchronize()
            step=version+1
            row=dict(step=step,ids=batchids,loss=[float(x) for x in losses],control_loss_maxabs=loss_delta,
                     head_gradient_norm=float(norm),encoding=encoding,backward=backward,commit=commit,
                     seconds=time.time()-stepstart)
            with (out/'train_steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            report.update(optimizer_updates=step,examples_seen=report['examples_seen']+len(indices))
            print(json.dumps({k:v for k,v in row.items() if k not in ['ids','encoding','backward','commit']}),flush=True)
            record()
            if step in [111,222]:
                saved=rng_state()
                tail=request('checkpoint',step)
                features=request('evaluate_features',step,ids=devids)
                dev.features_RGB_path=features['folder']
                dev.video_RGB_dict={i:(vid,str(Path(features['folder'])/'dev'/f'{vid}.pkl')) for i,vid in enumerate(devids)}
                metric=evaluate(native,args,model,loader,device,devids,out/f'eval_step{step:03d}')
                restore_rng(saved)
                report['evaluations'][str(step)]=metric
                path=out/f'head_checkpoint_step{step:03d}.pt'
                checkpoint=dict(model=model.state_dict(),rng=saved,next_batch_index=step,batches=batches,
                                config=vars(args),completed_steps=step,tail_checkpoint=tail,source_hashes=report['source_hashes'],
                                asset_hashes_digest=dict(train=report['train_assets_digest'],dev=report['dev_assets_digest']),
                                optimizer_state_saved=step==222)
                if step==222: checkpoint['optimizer']=optimizer.state_dict()
                torch.save(checkpoint,path)
                pair=dict(version=step,head_checkpoint=str(path),head_sha256=sha(path),tail=tail,
                          dev_features=features,joint_resumable=step==222)
                atomic_json(out/f'paired_checkpoint_step{step:03d}.json',pair)
                report['paired_checkpoints'][str(step)]=pair
                r1={d:metric['fusion'][d]['R1'] for d in ['T2V','V2T']}
                mean=sum(r1.values())/2
                if mean>selection['mean_R1'] and all(r1[d]>=initial_r1[d]-.5 for d in r1):
                    selection=dict(step=step,mean_R1=mean,R1=r1,**pair)
                    atomic_json(out/'selection.json',selection)
                record()
        report['worker_result']=request('finish',222)
        worker.stdin.close()
        assert worker.wait(timeout=10)==0
        strongest=json.loads((control/'selection.json').read_text())
        gain=selection['mean_R1']-strongest['mean_R1']
        guard=all(selection['R1'][d]>=strongest['R1'][d]-.5 for d in ['T2V','V2T'])
        report.update(status='completed',exit_status=0,selection=selection,control_selection=strongest,
                      selected_delta_mean_R1=gain,pilot_gate_passed=gain>=.5 and guard,
                      peak_head_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='Baseline tuning result only; no novel method, fresh confirmation or SOTA claim.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        if worker and worker.poll() is None: worker.stdin.close()
        record(True)
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__': main()
