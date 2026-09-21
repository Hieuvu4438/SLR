"""V2 SEDS loss-adaptation runner; reuse native training primitives and evaluator.

No TEST construction, extraction, corpus rehash or baseline replay. Existing
release scores are the step-0 anchor because this intervention changes loss only.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'shared'), str(ROOT/'research/slret_goal/tools')]
from extraction_resume import atomic_json
from inventory import sha
from seds_continue import rng_state, restore_rng
from seds_optimizer_precision import ensure_fp32_moments
from seds_runtime import eval_dataset, evaluate, model_inputs, native_kwargs, patch_pickle
from methods.seds_adaptation.objectives import fused_priority_loss
from methods.seds_adaptation.train_policies import configure_trainable_stage, training_modes
from methods.seds_adaptation.articulator_interaction import attach_articulator_interaction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--aux-weight', type=float, default=.25)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--policy', choices=['all','staged'], default='all')
    parser.add_argument('--fusion-steps', type=int, default=222)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--sign-lr', type=float)
    parser.add_argument('--interaction', choices=['none','product','additive'], default='none')
    cli = parser.parse_args()
    assert 1 <= cli.epochs <= 10 and 0 <= cli.aux_weight <= 2
    out = ROOT/'artifacts/slret_goal_v2'/cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    ledger = ROOT/'research/slret_goal_v2/EXPERIMENTS.jsonl'
    candidate = 'C03' if cli.interaction != 'none' else ('C02' if cli.policy == 'staged' else 'C01')
    report = dict(run_id=cli.run_id, candidate=candidate, status='running', pid=os.getpid(),
                  start_unix=started, command=sys.argv, config=vars(cli), steps=0,
                  selection_split='PH_adapted_DEV519' if not cli.smoke else 'none_smoke',
                  test_loaded=False, artifact_dir=str(out),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
                      Path(__file__), ROOT/'methods/seds_adaptation/objectives.py',
                      ROOT/'methods/seds_adaptation/train_policies.py',
                      ROOT/'methods/seds_adaptation/articulator_interaction.py',
                      ROOT/'research/slret_goal/tools/seds_runtime.py',
                      ROOT/'research/slret_goal/tools/seds_optimizer_precision.py']})

    def record(terminal=False):
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json', report)
        if terminal or report['steps'] == 0:
            with ledger.open('a') as f:
                f.write(json.dumps(report, allow_nan=False)+'\n')

    record()
    try:
        for relative in report['code_sha256']:
            destination = out/'source'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,destination)
        reserve = 15*1024**3
        planned = 128*1024**2 if cli.smoke else (2 if cli.policy == 'staged' else 4)*1024**3
        used = sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        assert used+planned <= 12*1024**3, 'V2 tranche storage allowance'
        assert shutil.disk_usage(out).free-planned >= reserve, 'Free disk reserve'
        torch.set_num_threads(4)
        base = ROOT/'third_party/SEDS'
        sys.path.insert(0, str(base))
        os.chdir(base)
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose, ph_pose_collate_fn
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose, ph_train_pose_collate_fn
        patch_pickle(ph_DataLoader_pose, ph_DataLoader_train_pose)
        reference_root = ROOT/'artifacts/slret_goal/seds-adapted-dev-eval-002'
        reference = json.loads((reference_root/'run.json').read_text())
        previous = json.loads((ROOT/'artifacts/slret_goal/seds-moment-control-001/run.json').read_text())
        args = argparse.Namespace(**previous['config'])
        args.output_dir, args.seed, args.epochs = str(out), cli.seed, cli.epochs
        if cli.lr is not None:
            args.lr = cli.lr
        if cli.sign_lr is not None:
            args.sign_lr = cli.sign_lr
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        assert not args.freeze_exfusion and not args.rgb_pose_kl
        train_root = ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        dev_root = ROOT/'artifacts/slret_goal/seds-adapted-dev-001'
        for root, split, count in [(train_root,'train',7096), (dev_root,'dev',519)]:
            manifest = json.loads((root/'run.json').read_text())
            assert manifest['status'] == 'completed' and manifest['completed'] == count
            assert (root/'labels'/f'{split}.pkl').is_file()
        args = native.set_seed_logger(args)
        device, n_gpu = native.init_device(args, args.local_rank)
        assert n_gpu == 1 and torch.distributed.get_world_size() == 1
        tokenizer = native.ClipTokenizer()
        train = ph_DataLoader_train_pose(**native_kwargs(args,tokenizer,train_root,'train'))
        dev, dev_ids = eval_dataset(ph_DataLoader_pose,args,tokenizer,dev_root)
        with (train_root/'labels/train.pkl').open('rb') as f:
            ids = list(pickle.load(f))
        assert len(ids) == 7096 and not set(ids)&set(dev_ids)
        loader = torch.utils.data.DataLoader(dev,batch_size=32,shuffle=False,num_workers=0,
                                             collate_fn=ph_pose_collate_fn)
        model = native.init_model(args,device)
        if cli.interaction != 'none':
            with torch.random.fork_rng():
                attach_articulator_interaction(model.signbert,512,32,cli.interaction)
        report.update(native_config=vars(args), hardware=torch.cuda.get_device_name(),
                      torch=torch.__version__, checkpoint=args.init_model,
                      checkpoint_sha256_inherited=previous['checkpoint_sha256'],
                      inherited_asset_digests={k:previous[k] for k in ['train_assets_digest','dev_assets_digest']},
                      data_validation='reuse V1 validated features; no full corpus rehash',
                      reference_report_sha256=sha(reference_root/'run.json'))
        initial = previous['evaluations']['0']
        initial_r1 = {d:initial['fusion'][d]['R1'] for d in ['T2V','V2T']}
        reference_mean = sum(initial_r1.values())/2
        incumbent_mean = previous['selection']['mean_R1']
        selection = dict(step=0,mean_R1=reference_mean,R1=initial_r1,checkpoint=args.init_model)
        report.update(reference_mean_R1=reference_mean,incumbent_mean_R1=incumbent_mean,
                      evaluations={'0':initial},step0='reused unchanged inference checkpoint; loss-only intervention')
        if cli.interaction != 'none':
            saved = rng_state()
            zero = evaluate(native,args,model,loader,device,dev_ids,out/'eval_step0000')
            for stream in ('fusion','pose','rgb'):
                actual = np.load(out/'eval_step0000'/f'{stream}_video_x_text.npy')
                anchor = np.load(reference_root/f'{stream}_video_x_text.npy')
                assert np.max(np.abs(actual-anchor)) <= 1e-4
                for direction in ('T2V','V2T'):
                    assert abs(zero[stream][direction]['R1']-initial[stream][direction]['R1']) < 1e-5
            restore_rng(saved)
            report['step0'] = 'zero-initialized interaction fullDEV score parity passed'
        elif cli.policy == 'staged':
            report['step0'] = 'inherited unchanged checkpoint/scorer; only trainable subset changes'
        atomic_json(out/'selection.json',selection)
        batches = []
        for epoch in range(cli.epochs):
            order = torch.randperm(len(ids),generator=torch.Generator().manual_seed(cli.seed+epoch)).tolist()
            batches.extend([order[i:i+32] for i in range(0,len(order),32)])
        schedule_steps = len(batches)
        if cli.policy == 'staged':
            assert 1 < cli.fusion_steps < schedule_steps
        if cli.smoke:
            batches = [list(range(32)) for _ in range(6 if cli.policy == 'staged' else 2)]
            if cli.policy == 'staged':
                assert cli.fusion_steps == 2, 'Stage smoke tests transition after two steps'
                schedule_steps = 6
        atomic_json(out/'batch_indices.json',batches)
        report['batch_order_sha256'] = sha(out/'batch_indices.json')
        active = configure_trainable_stage(model,'fusion') if cli.policy == 'staged' else None
        first_schedule = cli.fusion_steps if active is not None else schedule_steps
        optimizer, scheduler, wrapped = native.prep_optimizer(args,model,first_schedule,device,1,
                                                              args.local_rank,coef_lr=args.coef_lr)
        torch.cuda.reset_peak_memory_stats()
        tracked = dict(model.named_parameters())
        evaluation_steps = {111, *range(222,schedule_steps+1,222),schedule_steps}
        for step,indices in enumerate(batches,1):
            tick = time.time()
            if active is not None and step == cli.fusion_steps+1:
                del wrapped, optimizer
                active = configure_trainable_stage(model,'upper')
                optimizer,scheduler,wrapped = native.prep_optimizer(args,model,
                    schedule_steps-cli.fusion_steps,device,1,args.local_rank,coef_lr=args.coef_lr)
                report['stage_transition'] = dict(step=step,optimizer_reset=True,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            wrapped.train()
            if active is not None:
                training_modes(model,active)
            batch = ph_train_pose_collate_fn([train[i] for i in indices])
            optimizer.zero_grad()
            losses = wrapped(*model_inputs(batch,device))
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses)
            if step == 1:
                assert torch.allclose(fused_priority_loss(losses,1),losses[0],atol=1e-6,rtol=1e-6)
                report['native_objective_parity_on_real_batch'] = True
            loss = fused_priority_loss(losses,cli.aux_weight)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(),1.)
            assert torch.isfinite(grad_norm)
            ensure_fp32_moments(optimizer)
            check_update = step in (2,cli.fusion_steps+2) if active is not None else step == 2
            before = {n:p.detach().clone() for n,p in tracked.items()} if check_update else None
            optimizer.step()
            optimizer.zero_grad()
            torch.clamp_(model.clip.logit_scale.data,max=np.log(100))
            if before is not None:
                changed = [n for n,p in tracked.items() if not torch.equal(before[n],p)]
                assert changed, 'No parameter updated at step2'
                report['changed_parameter_tensors_step2'] = len(changed)
                report['changed_parameter_groups_step2'] = sorted({n.split('.')[0] for n in changed})
                report.setdefault('stage_update_checks',{})[str(step)] = dict(
                    changed=len(changed),groups=sorted({n.split('.')[0] for n in changed}))
                if active is not None:
                    assert all(tracked[n].requires_grad for n in changed), 'Frozen weight drift'
                if cli.interaction != 'none':
                    assert 'signbert.articulator_interaction.output.weight' in changed, 'Interaction did not activate'
                    report['interaction_output_updated'] = True
                del before
            torch.cuda.synchronize()
            row = dict(step=step,epoch=(step-1)//222,loss=float(loss),
                       native_components=[float(v) for v in losses],gradient_norm=float(grad_norm),
                       seconds=time.time()-tick)
            with (out/'train_steps.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            report['steps'] = step
            if not cli.smoke and step in evaluation_steps:
                saved = rng_state()
                metrics = evaluate(native,args,model,loader,device,dev_ids,out/f'eval_step{step:04d}')
                restore_rng(saved)
                report['evaluations'][str(step)] = metrics
                r1 = {d:metrics['fusion'][d]['R1'] for d in ['T2V','V2T']}
                mean = sum(r1.values())/2
                eligible = all(r1[d] >= initial_r1[d]-.5 for d in r1)
                print(json.dumps(dict(event='dev',step=step,mean_R1=mean,R1=r1,
                    delta_reference=mean-reference_mean,delta_incumbent=mean-incumbent_mean,
                    guardrail_pass=eligible)),flush=True)
                if mean > selection['mean_R1'] and eligible:
                    path = out/'best.pt'
                    torch.save(dict(model=model.state_dict(),step=step,config=vars(args),
                                    adaptation=vars(cli)),out/'best.tmp')
                    os.replace(out/'best.tmp',path)
                    selection = dict(step=step,mean_R1=mean,R1=r1,checkpoint=str(path),checkpoint_sha256=sha(path))
                    atomic_json(out/'selection.json',selection)
            if step % 10 == 0 or cli.smoke:
                print(json.dumps(row),flush=True)
            record()
        if not cli.smoke:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=None,
                            rng=rng_state(),batches=batches,next_batch_index=len(batches),
                            config=vars(args),adaptation=vars(cli),code_sha256=report['code_sha256'],
                            optimizer_precision='fp32_moments_native_parameters'),out/'last.pt')
        report.update(status='completed',exit_status=0,selection=selection,
                      delta_reference=selection['mean_R1']-reference_mean,
                      delta_incumbent=selection['mean_R1']-incumbent_mean,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='smoke_only' if cli.smoke else 'await_DEV_driven_refine_promote_drop')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record(True)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
