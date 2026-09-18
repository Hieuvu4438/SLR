"""Two full-batch joint steps with original-runtime I3D/native SEDS bridge."""
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
from extraction_resume import atomic_json
from seds_runtime import patch_pickle,native_kwargs,model_inputs,verify_assets
from seds_optimizer_precision import ensure_fp32_moments
from seds_continue import rng_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    started=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,optimizer_updates=0,
                test_loaded=False,dev_loaded=False,kind='baseline_finetuning_smoke_not_method',
                script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'research/slret_goal/RGB_TRAIN_SMOKE_PROTOCOL.md'),
                worker_sha256=sha(Path(__file__).with_name('rgb_bridge_worker.py')),steps=[])
    worker=None
    atomic_json(out/'run.json',report)
    try:
        allowance=128*1024**2
        assert shutil.disk_usage(out).free-allowance>15*1024**3
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())+allowance<15*1024**3
        torch.set_num_threads(4)
        sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'third_party/SEDS')]
        os.chdir(ROOT/'third_party/SEDS')
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose,ph_train_pose_collate_fn
        patch_pickle(ph_DataLoader_train_pose)
        control=ROOT/'artifacts/slret_goal/seds-moment-control-001'
        ctrl=json.loads((control/'run.json').read_text())
        args=argparse.Namespace(**ctrl['config'])
        args.output_dir,args.local_rank=str(out),int(os.environ.get('LOCAL_RANK',0))
        args=native.set_seed_logger(args)
        device,_=native.init_device(args,args.local_rank)
        root=ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        labels=pickle.load((root/'labels/train.pkl').open('rb'))
        batches=json.loads((control/'batch_indices.json').read_text())[:2]
        ids=list(labels)
        _,assets=verify_assets(root,'train',[ids[i] for batch in batches for i in batch])
        atomic_json(out/'asset_hashes.json',assets)
        data=ph_DataLoader_train_pose(**native_kwargs(args,native.ClipTokenizer(),root,'train'))
        model=native.init_model(args,device)
        expected=torch.load(args.init_model,map_location='cpu')
        assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in expected.items())
        del expected
        optimizer,_,wrapped=native.prep_optimizer(args,model,222,device,1,args.local_rank,coef_lr=args.coef_lr)
        report.update(config=vars(args),hardware=torch.cuda.get_device_name(),torch=torch.__version__,
                      control_sha256=sha(control/'run.json'),batch_indices=batches,
                      checkpoint_sha256=sha(args.init_model),
                      commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        with (out/'worker_stderr.log').open('x') as log:
            worker=subprocess.Popen(['/home/haipd/miniconda3/bin/python',str(Path(__file__).with_name('rgb_bridge_worker.py')),str(out)],
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
        reference=[json.loads(line) for line in (control/'train_steps.jsonl').read_text().splitlines()[:2]]
        torch.cuda.reset_peak_memory_stats()
        for version,indices in enumerate(batches):
            stepstart=time.time()
            wrapped.train()
            batch=ph_train_pose_collate_fn([data[i] for i in indices])
            batchids=[ids[i] for i in indices]
            encoding=request('encode',version,ids=batchids)
            assert sha(encoding['path'])==encoding['sha256']
            payload=torch.load(encoding['path'],map_location='cpu')
            assert payload['version']==version and payload['ids']==batchids
            rgb=payload['features'].to(device).detach().requires_grad_(True)
            delta=float((rgb.detach().cpu()-batch['RGB_feature']).abs().max())
            assert delta<=1e-4
            batch['RGB_feature']=rgb
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            optimizer.zero_grad()
            losses=wrapped(*model_inputs(batch,device))
            loss_delta=max(abs(float(a)-b) for a,b in zip(losses,reference[version]['loss']))
            assert loss_delta<=1e-7,('Matched loss',version,loss_delta)
            losses[0].backward()
            assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
            path=out/f'gradient_v{version}.pt'
            torch.save(dict(version=version,gradient=rgb.grad.detach().cpu()),path)
            backward=request('backward',version,sha256=sha(path))
            norm=torch.nn.utils.clip_grad_norm_(wrapped.parameters(),1.)
            assert torch.isfinite(norm)
            ensure_fp32_moments(optimizer)
            optimizer.step()
            optimizer.zero_grad()
            torch.clamp_(model.clip.logit_scale.data,max=np.log(100))
            changed=sum(not torch.equal(before[n],p) for n,p in model.named_parameters())
            del before
            assert (changed==0 if version==0 else changed>0)
            commit=request('commit',version)
            torch.cuda.synchronize()
            row=dict(step=version+1,ids=batchids,loss=[float(x) for x in losses],control_loss_maxabs=loss_delta,
                     input_feature_maxabs=delta,head_gradient_norm=float(norm),head_changed_tensors=changed,
                     encoding=encoding,backward=backward,commit=commit,seconds=time.time()-stepstart)
            report['steps'].append(row)
            report['optimizer_updates']=version+1
            atomic_json(out/'run.json',report)
            print(json.dumps(row),flush=True)
        report['worker_result']=request('finish',2)
        worker.stdin.close()
        assert worker.wait(timeout=10)==0
        torch.save(rng_state(),out/'head_rng_step002.pt')
        report.update(status='completed',exit_status=0,examples_seen=64,
                      peak_head_cuda_bytes=torch.cuda.max_memory_allocated(),
                      head_rng_sha256=sha(out/'head_rng_step002.pt'),resumable=False,
                      decision='Two-step activation only; no retrieval gain or full pilot admission.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        if worker and worker.poll() is None:
            # Closing the pipe asks an idle worker to end; timeout owns hard cancellation.
            worker.stdin.close()
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__': main()
