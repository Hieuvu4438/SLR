"""Bounded native SEDS continuation control, explicit TRAIN/DEV, no TEST."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json, digest
from seds_runtime import eval_dataset, evaluate, model_inputs, native_kwargs, patch_pickle, verify_assets
from seds_optimizer_precision import ensure_fp32_moments


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--dev-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--fp32-moments', action='store_true')
    cli = parser.parse_args()
    cli.train_root, cli.dev_root = cli.train_root.resolve(), cli.dev_root.resolve()
    out = ROOT / 'artifacts/slret_goal' / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    report = dict(run_id=cli.run_id, status='running', pid=os.getpid(), command=sys.argv,
                  seed=42, kind='baseline_continuation_control_not_method', test_loaded=False,
                  selection_split='none_smoke' if cli.smoke else 'historically_exposed_PH_dev',
                  optimizer_updates=0, examples_seen=0, batch_size=32,
                  script_sha256=sha(__file__), runtime_sha256=sha(Path(__file__).with_name('seds_runtime.py')),
                  protocol_sha256=sha(ROOT / ('research/slret_goal/SEDS_MOMENT_CONTROL_PROTOCOL.md' if cli.fp32_moments else 'research/slret_goal/SEDS_CONTINUATION_PROTOCOL.md')),
                  optimizer_precision='fp32_moments_native_parameters' if cli.fp32_moments else 'native',
                  precision_helper_sha256=sha(Path(__file__).with_name('seds_optimizer_precision.py')),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=ROOT)).hexdigest())
    def record(ledger=False):
        report['wall_seconds'] = time.time()-start
        atomic_json(out / 'run.json', report)
        if ledger:
            with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
                f.write(json.dumps(report) + '\n')
    record(True)
    try:
        # Fail before initializing NCCL/model or doing step0 evaluation.
        reserve = 15*1024**3
        checkpoint_bytes = (ROOT/'third_party/SEDS/ckpt/ph_best_model.bin').stat().st_size
        planned_upper = checkpoint_bytes*3*(1 if cli.smoke else 2)
        if cli.fp32_moments and cli.smoke:
            planned_upper = 256*1024**2  # no checkpoint in numerical-control smoke
        assert shutil.disk_usage(out).free-planned_upper >= reserve, 'Prelaunch disk reserve'
        torch.set_num_threads(4)
        sys.path.insert(0, str(ROOT / 'shared'))
        base = ROOT / 'third_party/SEDS'
        sys.path.insert(0, str(base))
        os.chdir(base)
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose, ph_pose_collate_fn
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose, ph_train_pose_collate_fn
        patch_pickle(ph_DataLoader_pose, ph_DataLoader_train_pose)
        # Reuse the runtime-verified release recipe, with explicit control deltas.
        reference_root = ROOT / 'artifacts/slret_goal/seds-adapted-dev-eval-002'
        reference = json.loads((reference_root / 'run.json').read_text())
        args = argparse.Namespace(**reference['config'])
        args.do_train, args.do_eval = True, False
        args.batch_size, args.batch_size_val, args.epochs = 32, 32, 1
        args.output_dir, args.seed = str(out), 42
        args.gradient_accumulation_steps, args.num_thread_reader = 1, 0
        args.local_rank, args.distributed = int(os.environ.get('LOCAL_RANK',0)), True
        args.data_path, args.features_path = str(cli.train_root / 'labels'), str(cli.train_root / 'pose')
        args.features_RGB_path = str(cli.train_root / 'rgb')
        args = native.set_seed_logger(args)
        device, n_gpu = native.init_device(args, args.local_rank)
        assert n_gpu == 1 and torch.distributed.get_world_size() == 1
        labels = pickle.load((cli.train_root / 'labels/train.pkl').open('rb'))
        original_train = pickle.load((base / 'data_ph/train.pkl').open('rb'))
        assert list(labels) == list(original_train), 'Training must keep canonical order/complete labels'
        ids = list(labels)
        selected_ids = ids[:32] if cli.smoke else ids
        train_report, train_hashes = verify_assets(cli.train_root, 'train', selected_ids, not cli.smoke)
        tokenizer = native.ClipTokenizer()
        train = ph_DataLoader_train_pose(**native_kwargs(args, tokenizer, cli.train_root, 'train'))
        dev, dev_ids = eval_dataset(ph_DataLoader_pose, args, tokenizer, cli.dev_root)
        assert not set(ids) & set(dev_ids)
        dev_report, dev_hashes = verify_assets(cli.dev_root, 'dev', dev_ids)
        for key in ['pose_checkpoint_sha256','pose_config_sha256','rgb_checkpoint_sha256',
                    'rgb_source_sha256','rgb_recipe_sha256','native_loader_sha256']:
            assert train_report[key] == dev_report[key], ('Preprocessing mismatch',key)
        report.update(config=vars(args), hardware=torch.cuda.get_device_name(), torch=torch.__version__,
                      checkpoint_sha256=sha(args.init_model), reference_run_sha256=sha(reference_root/'run.json'),
                      train_assets_digest=digest(train_hashes), dev_assets_digest=digest(dev_hashes))
        atomic_json(out / 'asset_hashes.json', dict(train=train_hashes, dev=dev_hashes))
        model = native.init_model(args, device)
        expected = torch.load(args.init_model, map_location='cpu')
        state = model.state_dict()
        assert set(expected) == set(state)
        assert all(torch.equal(state[k].cpu(),v) for k,v in expected.items())
        del expected, state
        report['checkpoint_tensor_parity'] = 'exact'
        torch.cuda.reset_peak_memory_stats()
        if cli.smoke:
            batches = [list(range(32)), list(range(32))]
        else:
            order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(42)).tolist()
            batches = [order[i:i+32] for i in range(0,len(order),32)]
            assert len(batches) == 222 and len(batches[-1]) == 24
        atomic_json(out / 'batch_indices.json', batches)
        report['batch_order_sha256'] = sha(out / 'batch_indices.json')
        data_loader = torch.utils.data.DataLoader(dev, batch_size=32, shuffle=False,
                                                num_workers=0, collate_fn=ph_pose_collate_fn)
        # Validate new training wrapper at step0, rather than assume identical wiring.
        initial_rng = rng_state()
        initial = evaluate(native,args,model,data_loader,device,dev_ids,out/'eval_step000')
        for stream in ['fusion','pose','rgb']:
            a=np.load(out/'eval_step000'/(stream+'_video_x_text.npy'))
            b=np.load(reference_root/(stream+'_video_x_text.npy'))
            assert np.max(np.abs(a-b)) <= 1e-4, ('Step0 score parity',stream)
            ref=json.loads((reference_root/(stream+'_metrics.json')).read_text())
            for d in ['T2V','V2T']:
                for key in ['R1','R5','R10','MeanR']:
                    assert abs(initial[stream][d][key]-ref[d][key]) <= 1e-5
        restore_rng(initial_rng)
        report['step0_parity'] = 'passed'
        initial_r1 = {d:initial['fusion'][d]['R1'] for d in ['T2V','V2T']}
        selection = dict(step=0, mean_R1=sum(initial_r1.values())/2, R1=initial_r1,
                         checkpoint=args.init_model, checkpoint_sha256=report['checkpoint_sha256'])
        report['evaluations'] = {'0':initial}
        atomic_json(out/'selection.json',selection)
        optimizer, scheduler, wrapped = native.prep_optimizer(args,model,222,device,1,args.local_rank,coef_lr=args.coef_lr)
        source_files = [base/'main_task_retrieval.py',base/'modules/modeling.py',
                        base/'modules/optimization.py',base/'dataloaders/dataloader_ph_retrieval_train_pose.py']
        report['source_sha256'] = {str(p):sha(p) for p in source_files}
        total_tensor_bytes = sum(p.numel()*p.element_size() for p in model.state_dict().values())
        # Model + two Adam states; two periodic files for full, one for smoke.
        planned = total_tensor_bytes * 3 * (1 if cli.smoke else 2)
        if cli.fp32_moments:
            # Step111 model only; step222 model plus FP32 Adam states. Smoke: none.
            planned = 0 if cli.smoke else 2*total_tensor_bytes + 8*sum(p.numel() for p in model.parameters())
            planned += 256*1024**2  # metrics/serialization overhead
        campaign = ROOT/'artifacts/slret_goal'
        current_bytes = sum(p.stat().st_size for p in campaign.rglob('*') if p.is_file())
        assert shutil.disk_usage(out).free - planned >= 15*1024**3, 'Disk reserve'
        assert current_bytes+planned <= 15*1024**3, 'Campaign artifact cap'
        tracked = dict(model.named_parameters())
        changed_steps = []
        for index, indices in enumerate(batches):
            step_start=time.time()
            wrapped.train()
            batch=ph_train_pose_collate_fn([train[i] for i in indices])
            before = {n:p.detach().clone() for n,p in tracked.items()} if cli.smoke else None
            optimizer.zero_grad()
            losses=wrapped(*model_inputs(batch,device))
            # Disabled native losses are Python 0.0, not tensors.
            assert all(torch.isfinite(torch.as_tensor(x)).all() for x in losses)
            losses[0].backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(wrapped.parameters(),1.0)
            assert torch.isfinite(grad_norm)
            if cli.fp32_moments:
                ensure_fp32_moments(optimizer)
            optimizer.step()
            optimizer.zero_grad()
            torch.clamp_(model.clip.logit_scale.data,max=np.log(100))
            changed = None
            if before is not None:
                changed=sum(not torch.equal(before[n],p) for n,p in tracked.items())
                changed_steps.append(changed)
                del before
            torch.cuda.synchronize()
            step=index+1
            item=dict(step=step, ids=[ids[i] for i in indices], loss=[float(x) for x in losses],
                      gradient_norm=float(grad_norm), changed_parameter_tensors=changed,
                      seconds=time.time()-step_start)
            with (out/'train_steps.jsonl').open('a') as f:
                f.write(json.dumps(item)+'\n')
            report.update(optimizer_updates=step,examples_seen=report['examples_seen']+len(indices))
            if step in ([2] if cli.smoke else [111,222]):
                if not cli.smoke:
                    saved_rng=rng_state()
                    metric=evaluate(native,args,model,data_loader,device,dev_ids,out/f'eval_step{step:03d}')
                    restore_rng(saved_rng)
                    report['evaluations'][str(step)]=metric
                path=out/f'checkpoint_step{step:03d}.pt'
                checkpoint_payload=dict(model=model.state_dict(),rng=rng_state(),
                                next_batch_index=step,batches=batches,config=vars(args),
                                script_sha256=report['script_sha256'],runtime_sha256=report['runtime_sha256'],
                                asset_hashes_digest=dict(train=report['train_assets_digest'],dev=report['dev_assets_digest']),
                                completed_steps=step,optimizer_precision=report['optimizer_precision'])
                resumable = not cli.fp32_moments or (not cli.smoke and step==222)
                if resumable:
                    checkpoint_payload['optimizer']=optimizer.state_dict()
                checkpoint_payload['optimizer_state_saved']=resumable
                if not (cli.fp32_moments and cli.smoke):
                    torch.save(checkpoint_payload,path)
                if not cli.smoke:
                    r1={d:metric['fusion'][d]['R1'] for d in ['T2V','V2T']}
                    mean=sum(r1.values())/2
                    if mean>selection['mean_R1'] and all(r1[d]>=initial_r1[d]-.5 for d in r1):
                        selection=dict(step=step,mean_R1=mean,R1=r1,checkpoint=str(path),checkpoint_sha256=sha(path))
                        atomic_json(out/'selection.json',selection)
            if step%10 == 0 or cli.smoke:
                print(json.dumps({k:v for k,v in item.items() if k!='ids'}),flush=True)
            record()
        if cli.smoke:
            assert changed_steps[-1] > 0, 'No actual parameter update after warmup'
            report['changed_parameter_tensors_by_step']=changed_steps
        if cli.fp32_moments:
            states=[optimizer.state[p] for group in optimizer.param_groups for p in group['params']
                    if p.dtype==torch.float16 and p in optimizer.state and optimizer.state[p]]
            assert all(s['next_m'].dtype==s['next_v'].dtype==torch.float32 for s in states)
            report['fp16_parameter_fp32_moment_states']=len(states)
            report['nonzero_m_zero_v_count']=sum(int(((s['next_m']!=0)&(s['next_v']==0)).sum()) for s in states)
        report.update(status='completed',exit_status=0,selection=selection,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='Smoke only; start full run independently from release.' if cli.smoke else
                      'Baseline control only; any continuation improvement is not a method gain.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record(True)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
