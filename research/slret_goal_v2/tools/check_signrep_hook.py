"""Bounded changed-path check on the first DEV batch, not a baseline campaign."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'shared'),str(ROOT/'research/slret_goal/tools'),str(ROOT/'third_party/SEDS')]
from extraction_resume import atomic_json
from seds_runtime import eval_dataset, model_inputs, patch_pickle
from methods.seds_adaptation.masked_pose import attach_masked_pose
from methods.seds_adaptation.signrep_transfer import attach_signrep_transfer


def main():
    out = ROOT/'artifacts/slret_goal_v2/signrep-hook-check-002'
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(status='running',pid=os.getpid(),test_used=False,training=False)
    try:
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose,ph_pose_collate_fn
        torch.set_num_threads(4)
        args = argparse.Namespace(**json.loads((ROOT/'artifacts/slret_goal_v2/seds-signrep-transfer-smoke-001/run.json').read_text())['native_config'])
        args.output_dir = str(out)
        os.chdir(ROOT/'third_party/SEDS')
        args = native.set_seed_logger(args)
        device,_ = native.init_device(args,0)
        patch_pickle(ph_DataLoader_pose)
        dev,ids = eval_dataset(ph_DataLoader_pose,args,native.ClipTokenizer(),ROOT/'artifacts/slret_goal/seds-adapted-dev-001')
        model = native.init_model(args,device)
        with torch.random.fork_rng():
            attach_masked_pose(model,control=True,seed=42)
        model.eval()
        batch = ph_pose_collate_fn([dev[i] for i in range(32)])
        batch = {k:v.to(device) for k,v in batch.items()}
        inputs = ({'pose':batch['right_pose']},{'pose':batch['left_pose']},
                  dict(pose=batch['body_pose'],clips_start=batch['body_clips_start'],
                       mask=batch['body_mask'],rgb=batch['RGB_feature']))
        original = model.get_sign_output
        with torch.no_grad():
            before = model.get_visual_output(*inputs)
            repeat = model.get_visual_output(*inputs)
            with torch.random.fork_rng():
                attach_signrep_transfer(model)
            model.eval()
            hooked = model.get_visual_output(*inputs)
            model.get_sign_output = original
            unhooked = model.get_visual_output(*inputs)
        report['comparisons'] = {}
        for label,left,right in [('native_repeat',before,repeat),('allocation_and_hook',repeat,hooked),('hook_only',hooked,unhooked)]:
            report['comparisons'][label] = [dict(shape=list(a.shape),exact=torch.equal(a,b),max_diff=float((a.float()-b.float()).abs().max())) for a,b in zip(left,right)]
        report.update(status='completed',dev_ids=ids[:32],hook_exact=all(torch.equal(a,b) for a,b in zip(hooked,unhooked)))
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json',report)
        print(json.dumps(report),flush=True)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
