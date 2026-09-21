"""Evaluate the already-trained step160 delta. No optimizer or training replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'shared'),str(ROOT/'research/slret_goal/tools')]
from extraction_resume import atomic_json
from inventory import sha

SOURCE = ROOT/'artifacts/slret_goal_v2/seds-fused-dcl-blend005-offload-001'
RUN = 'seds-fused-dcl-blend005-final-eval-001'
LAST_SHA = '995d749e5e2c769b4f989ca772cbbeb2ed4047591884d812c637651def4c1ce9'
BEST_SHA = 'e152c297f2fa50ee3aabe9d9ad4dc8ba14116dd86fd8e4831a9f4a12af804e32'


def check_checkpoint(checkpoint,source_report):
    if checkpoint.get('next_batch_index') != 160 or len(checkpoint.get('batches',[])) != 160:
        raise ValueError('Checkpoint does not contain all160 completed updates')
    if checkpoint.get('checkpoint_format') != 'GCN_fusion_delta_requires_native_release':
        raise ValueError('Unexpected checkpoint format')
    if checkpoint['adaptation'] != source_report['config']:
        raise ValueError('Saved experiment recipe differs from source report')
    if checkpoint['code_sha256'] != source_report['code_sha256']:
        raise ValueError('Saved code IDs differ from source report')
    for key in ('masked_update_checks_passed','delta_roundtrip_passed','dcl_loss_replacement_passed'):
        if not source_report.get(key):
            raise ValueError('Missing training gate: '+key)


def choose_selection(initial,previous,final,last_path,last_sha):
    """Keep the original0/80/160 selector/guardrails, including ties."""
    r1 = {d:final['fusion'][d]['R1'] for d in ('T2V','V2T')}
    mean = sum(r1.values())/2
    eligible = all(r1[d] >= initial['fusion'][d]['R1']-.5 for d in r1)
    if eligible and mean > previous['mean_R1']+1e-8:
        return dict(step=160,mean_R1=mean,R1=r1,checkpoint=str(last_path),
                    checkpoint_sha256=last_sha),mean
    return dict(previous),mean


def main():
    import torch
    from seds_runtime import eval_dataset,evaluate,patch_pickle
    from seds_continue import restore_rng
    from methods.seds_adaptation.signrep_transfer import load_transfer_state

    out = ROOT/'artifacts/slret_goal_v2'/RUN
    out.mkdir(parents=True,exist_ok=False)
    started = time.time()
    report = dict(run_id=RUN,status='running',pid=os.getpid(),steps=0,
        operation='final_DEV_evaluation_only',training_steps=160,new_training_steps=0,
        start_unix=started,source_run=SOURCE.name,source_status='TIMED_OUT',test_loaded=False,
        checkpoint=str(SOURCE/'last.pt'),checkpoint_sha256=LAST_SHA)
    def record():
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
    record()
    try:
        reserve=128*1024**2
        used=sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        if used+reserve>40*1024**3 or shutil.disk_usage(out).free-reserve<15*1024**3:
            raise ValueError('Evaluation storage reservation unavailable')
        old=json.loads((SOURCE/'run.json').read_text())
        previous=json.loads((SOURCE/'selection.json').read_text())
        if sha(SOURCE/'last.pt')!=LAST_SHA or sha(SOURCE/'best.pt')!=BEST_SHA:
            raise ValueError('Recovery checkpoint changed')
        if previous.get('checkpoint_sha256')!=BEST_SHA or previous['step']!=80:
            raise ValueError('Unexpected existing selection')
        checkpoint=torch.load(SOURCE/'last.pt',map_location='cpu')
        check_checkpoint(checkpoint,old)
        rows=[json.loads(x) for x in (SOURCE/'train_steps.jsonl').read_text().splitlines()]
        if [x['step'] for x in rows]!=list(range(1,161)):
            raise ValueError('Training log does not confirm all160 updates')
        for relative in ('research/slret_goal/tools/seds_runtime.py','third_party/SEDS/main_task_retrieval.py',
                         'third_party/SEDS/modules/modeling_gcn.py','third_party/SEDS/modules/modeling_graph.py'):
            if sha(ROOT/relative)!=checkpoint['code_sha256'][relative]:
                raise ValueError('Relevant native evaluation code changed: '+relative)
        sources=[Path(__file__),ROOT/'research/slret_goal/tools/seds_runtime.py',
                 ROOT/'research/slret_goal/tools/seds_continue.py',
                 ROOT/'methods/seds_adaptation/signrep_transfer.py',
                 ROOT/'third_party/SEDS/main_task_retrieval.py',ROOT/'third_party/SEDS/modules/modeling.py']
        report.update(code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
            source_report_sha256=sha(SOURCE/'run.json'),saved_training_code_sha256=checkpoint['code_sha256'],
            source_checkpoint_step_verified=True)
        for p in sources:
            target=out/'source'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
        torch.set_num_threads(4)
        args=argparse.Namespace(**checkpoint['config'])
        args.output_dir=str(out)
        args.local_rank=int(os.environ.get('LOCAL_RANK',0))
        base=checkpoint['base_checkpoint']
        args.init_model=base['path']
        if sha(Path(base['path']))!=base['sha256']:
            raise ValueError('Native base checkpoint changed')
        report['base_checkpoint']=base
        sys.path.insert(0,str(ROOT/'third_party/SEDS'))
        os.chdir(ROOT/'third_party/SEDS')
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose,ph_pose_collate_fn
        patch_pickle(ph_DataLoader_pose)
        args=native.set_seed_logger(args)
        device,n_gpu=native.init_device(args,args.local_rank)
        assert n_gpu==1 and torch.distributed.get_world_size()==1
        total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(16*1024**3/total,device)
        dev,ids=eval_dataset(ph_DataLoader_pose,args,native.ClipTokenizer(),
                           ROOT/'artifacts/slret_goal/seds-adapted-dev-001')
        if ids!=old['subset_data']['dev_ids']:
            raise ValueError('DEV identity/order differs from training evaluation')
        loader=torch.utils.data.DataLoader(dev,batch_size=32,shuffle=False,num_workers=0,
                                           collate_fn=ph_pose_collate_fn)
        model=native.init_model(args,device)
        load_transfer_state(model,checkpoint['model'])
        if any(not torch.equal(model.state_dict()[k].cpu(),v) for k,v in checkpoint['model'].items()):
            raise ValueError('Delta load tensor equality failed')
        report.update(delta_load_exact=True,dev_ids=ids,config=vars(args),
                      stage='evaluate_final160',torch=torch.__version__,hardware=torch.cuda.get_device_name())
        model.eval()
        model.requires_grad_(False)
        restore_rng(checkpoint['rng'])
        del checkpoint
        record()
        with torch.no_grad():
            metrics=evaluate(native,args,model,loader,device,ids,out/'eval_step0160')
        selection,mean=choose_selection(old['evaluations']['0'],previous,metrics,SOURCE/'last.pt',LAST_SHA)
        report.update(status='completed',steps=1,stage='finished',exit_status=0,
            evaluations={**old['evaluations'],'160':metrics},selection=selection,final_mean_R1=mean,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            decision='await user; recovered evaluation, original timeout record retained')
        atomic_json(out/'selection.json',selection)
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        record()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
