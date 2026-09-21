"""C08 fixed smoke -> masked reconstruction -> matched clean continuation.

Fail closed on implementation errors; no retry, tuning, TEST or next family.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/slret_goal/tools'))
from extraction_resume import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',choices=['001','002'],default='001')
    parser.add_argument('--replicate-control',action='store_true',
                        help='Registered clean GCN recipe, seeds1337/2026; no masking or new tuning')
    parser.add_argument('--fusion-ablation',action='store_true')
    parser.add_argument('--clip-temporal-pilot',action='store_true')
    parser.add_argument('--long-gcn-pilot',action='store_true')
    parser.add_argument('--long-fusion-control',action='store_true')
    parser.add_argument('--long-gcn-replication',action='store_true')
    parser.add_argument('--freeze-gcn-pilot',action='store_true')
    args=parser.parse_args()
    if sum((args.fusion_ablation,args.replicate_control,args.clip_temporal_pilot,args.long_gcn_pilot,args.long_fusion_control,args.long_gcn_replication,args.freeze_gcn_pilot)) > 1:
        raise ValueError('Choose only one registered chain')
    long_horizon=args.long_gcn_pilot or args.long_fusion_control or args.long_gcn_replication or args.freeze_gcn_pilot
    out=ROOT/f'artifacts/slret_goal_v2/c08-masked-pair-{args.attempt}'
    jobs=[(f'seds-masked-smoke-{args.attempt}','reconstruct',240,True),
          (f'seds-masked-pose-{args.attempt}','reconstruct',1080,False),
          (f'seds-masked-control-{args.attempt}','control',1080,False)]
    jobs=[(*job,42) for job in jobs]
    if args.replicate_control:
        out=ROOT/'artifacts/slret_goal_v2/gcn-clean-replication-001'
        jobs=[(f'seds-gcn-clean-seed{seed}-001','control',900,False,seed) for seed in (1337,2026)]
    if args.fusion_ablation:
        out=ROOT/'artifacts/slret_goal_v2/gcn-fusion-ablation-001'
        jobs=[(f'seds-gcn-fusion-ablation-seed{seed}-001','control',450,False,seed) for seed in (42,1337,2026)]
    if args.clip_temporal_pilot:
        out=ROOT/'artifacts/slret_goal_v2/c12-gcn-clip-temporal-001'
        jobs=[('seds-gcn-clip-temporal-smoke-001','control',60,True,42),
              ('seds-gcn-clip-temporal-001','control',800,False,42)]
    if args.long_gcn_pilot:
        out=ROOT/'artifacts/slret_goal_v2/gcn-horizon3-pilot-001'
        # Native three-epoch batching/schedule already supported. Reuse clean
        # GCN smoke; actual first-two-update gates also run inside this pilot.
        jobs=[('seds-gcn-horizon3-001','control',2100,False,42)]
    if args.long_fusion_control:
        out=ROOT/'artifacts/slret_goal_v2/gcn-horizon3-control-001'
        jobs=[('seds-gcn-horizon3-fusion-control-001','control',800,False,42)]
    if args.long_gcn_replication:
        out=ROOT/'artifacts/slret_goal_v2/gcn-horizon3-replication-001'
        jobs=[(f'seds-gcn-horizon3-seed{seed}-001','control',2100,False,seed) for seed in (1337,2026)]
    if args.freeze_gcn_pilot:
        out=ROOT/'artifacts/slret_goal_v2/gcn-freeze-pilot-001'
        jobs=[('seds-gcn-freeze-smoke-001','control',60,True,42),
              ('seds-gcn-freeze-001','control',1300,False,42)]
    for name,_,_,_,_ in jobs:
        if (out.parent/name).exists(): raise FileExistsError(name)
    out.mkdir(exist_ok=False)
    sources=[Path(__file__), ROOT/'research/slret_goal_v2/tools/train_seds_extended.py',
             *sorted((ROOT/'methods/seds_adaptation').glob('*.py')),
             ROOT/'research/slret_goal/tools/seds_runtime.py', ROOT/'third_party/SEDS/main_task_retrieval.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    started=time.time()
    report=dict(run_id=out.name,status='running',pid=os.getpid(),completed=[],test_used=False,
                replicate_control=args.replicate_control,fusion_ablation=args.fusion_ablation,
                clip_temporal_pilot=args.clip_temporal_pilot,long_gcn_pilot=args.long_gcn_pilot,
                long_fusion_control=args.long_fusion_control,
                long_gcn_replication=args.long_gcn_replication,
                freeze_gcn_pilot=args.freeze_gcn_pilot,
                source_sha256=hashes)
    child=None
    def record():
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
    def terminate(signum,frame):
        if child is not None and child.poll() is None: child.terminate()
        raise TimeoutError('Parent received SIGTERM')
    signal.signal(signal.SIGTERM,terminate)
    record()
    try:
        for name,mode,seconds,smoke,seed in jobs:
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
                raise ValueError('Source drift across registered pair')
            command=['timeout','--signal=TERM','--kill-after=10s',f'{seconds}s',
                '/home/haipd/miniconda3/envs/seds/bin/python','-m','torch.distributed.run',
                '--standalone','--nproc_per_node=1',str(ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
                '--run-id',name,'--masked-pose',mode,'--policy','fusion','--aux-weight','1',
                '--batch-size','32','--seed',str(seed),'--lr','1e-5','--sign-lr','1e-6',
                '--epochs','3' if long_horizon else '1','--early-stop-drop-pp','2']
            if smoke: command.append('--smoke')
            if args.fusion_ablation or args.long_fusion_control: command.append('--clean-fusion-control')
            if args.clip_temporal_pilot: command.append('--gcn-clip-temporal')
            if long_horizon: command.append('--gcn-long-horizon')
            if args.freeze_gcn_pilot: command.extend(['--gcn-freeze-after','2' if smoke else '222'])
            with (out/(name+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                report['current']=dict(run_id=name,pid=child.pid,command=command);record()
                code=child.wait()
            path=out.parent/name/'run.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            report['completed'].append(dict(run_id=name,returncode=code,status=result.get('status'),
                wall_seconds=result.get('wall_seconds'),steps=result.get('steps'),selection=result.get('selection')))
            record()
            if code or result.get('status')!='completed': raise RuntimeError(f'{name} failed; no automatic retry')
            if not result.get('masked_update_checks_passed'): raise ValueError('Update/buffer checks failed')
            if mode=='reconstruct' and not result.get('reconstruction_encoder_gradient_passed'):
                raise ValueError('Auxiliary did not reach encoder')
            expected={'encoder':1e-6,'fusion':1e-5}
            if args.fusion_ablation or args.long_fusion_control: expected={'fusion':1e-5}
            if args.clip_temporal_pilot:
                expected['clip_temporal']=1e-6
                if not result.get('clip_temporal_updated'): raise ValueError('Clip temporal adaptation inactive')
            if mode=='reconstruct': expected['decoder']=1e-4
            if {g['adaptation_group']:g['lr'] for g in result['actual_optimizer_groups']} != expected:
                raise ValueError('LR mismatch')
            if args.freeze_gcn_pilot and not result.get('stopped_early'):
                if not result.get('frozen_gcn_unchanged_final') or not result.get('only_fusion_updated'):
                    raise ValueError('GCN freeze/fusion continuation gates failed')
                if not smoke:
                    matched=json.loads((out.parent/'seds-gcn-horizon3-001/run.json').read_text())
                    for step in ('111','222'):
                        if result['evaluations'][step]['fusion']!=matched['evaluations'][step]['fusion']:
                            raise ValueError('Pre-freeze retrieval differs from matched R1 control')
                    report['pre_freeze_metrics_match']=True
            if long_horizon and not smoke:
                anchor_name='seds-gcn-horizon3-001' if args.long_gcn_replication else 'seds-masked-control-002'
                anchor=json.loads((out.parent/anchor_name/'run.json').read_text())
                for key in ('masked_pose','policy','aux_weight','batch_size','lr','sign_lr',
                            'seed','early_stop_drop_pp','interaction','label_smoothing'):
                    if key=='seed' and args.long_gcn_replication:continue
                    if result['config'][key]!=anchor['config'][key]:
                        raise ValueError(f'Horizon refinement changed other recipe field: {key}')
                if result['config']['epochs']!=3:
                    raise ValueError('Unregistered horizon')
                if result['config']['seed']!=seed:
                    raise ValueError('Unregistered seed')
                if result['checkpoint_sha256_inherited']!=anchor['checkpoint_sha256_inherited'] or result['inherited_asset_digests']!=anchor['inherited_asset_digests']:
                    raise ValueError('Horizon refinement base/data mismatch')
                batches=json.loads((out.parent/name/'batch_indices.json').read_text())
                first_name=f'seds-gcn-clean-seed{seed}-001' if args.long_gcn_replication else 'seds-masked-control-002'
                first=json.loads((out.parent/first_name/'batch_indices.json').read_text())
                if len(batches)!=666 or batches[:len(first)]!=first:
                    raise ValueError('Horizon refinement first-epoch order mismatch')
                report['horizon_refinement_checks_passed']=True
                if args.long_fusion_control:
                    paired=json.loads((out.parent/'seds-gcn-horizon3-001/run.json').read_text())
                    if not result.get('only_fusion_updated') or result['batch_order_sha256']!=paired['batch_order_sha256']:
                        raise ValueError('Three-epoch fusion control update/order mismatch')
                    if result['config']['early_stop_drop_pp']!=paired['config']['early_stop_drop_pp']:
                        raise ValueError('Three-epoch stopping-rule mismatch')
                    report['selected_gcn_minus_fusion']=paired['selection']['mean_R1']-result['selection']['mean_R1']
            if args.replicate_control or args.fusion_ablation or (args.clip_temporal_pilot and not smoke):
                anchor_name='seds-masked-control-002' if seed==42 or args.replicate_control else f'seds-gcn-clean-seed{seed}-001'
                anchor=json.loads((out.parent/anchor_name/'run.json').read_text())
                for key in ('masked_pose','policy','aux_weight','batch_size','lr','sign_lr',
                            'epochs','early_stop_drop_pp','interaction','label_smoothing'):
                    if result['config'][key]!=anchor['config'][key]:
                        raise ValueError(f'Replication recipe mismatch: {key}')
                if result['checkpoint_sha256_inherited']!=anchor['checkpoint_sha256_inherited'] or result['inherited_asset_digests']!=anchor['inherited_asset_digests']:
                    raise ValueError('Replication base/data mismatch')
                if args.fusion_ablation or args.clip_temporal_pilot:
                    if (args.fusion_ablation and not result.get('only_fusion_updated')) or result['batch_order_sha256']!=anchor['batch_order_sha256']:
                        raise ValueError('Ablation update/batch mismatch')
        if not (args.replicate_control or args.fusion_ablation or args.clip_temporal_pilot or long_horizon):
            treatment=json.loads((out.parent/jobs[1][0]/'run.json').read_text())
            control=json.loads((out.parent/jobs[2][0]/'run.json').read_text())
            if treatment['batch_order_sha256']!=control['batch_order_sha256']:
                raise ValueError('Batch order mismatch')
            report['selected_treatment_minus_control']=treatment['selection']['mean_R1']-control['selection']['mean_R1']
        report.update(status='completed',exit_status=0,
            decision='await_user_result_review; no automatic further experiments')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc()); raise
    finally: record()


if __name__=='__main__': main()
