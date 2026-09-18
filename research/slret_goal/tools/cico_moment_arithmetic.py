"""One real TRAIN gradient; isolated optimizer arithmetic, no model updates."""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback

import torch
import yaml

from adamw_arithmetic import isolated_step,normalized_update,squares
from extraction_resume import atomic_json
from inventory import ROOT,sha


def tensor_digest(values):
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        h.update(name.encode())
        h.update(str((value.dtype,tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dataset',choices=['ph','csl'],required=True)
    parser.add_argument('--batch-size',type=int,choices=[32,512],default=32)
    cli=parser.parse_args()
    out=ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    start=time.time()
    report=dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
        dataset=cli.dataset,seed=42,batch_size=cli.batch_size,optimizer_updates=0,dev_loaded=False,
        test_loaded=False,script_sha256=sha(__file__),arithmetic_source_sha256=sha(ROOT/'research/slret_goal/tools/adamw_arithmetic.py'),
        protocol_sha256=sha(ROOT/'research/slret_goal/CICO_MOMENT_ARITHMETIC_PROTOCOL.md'))
    atomic_json(out/'run.json',report)
    try:
        assert shutil.disk_usage(out).free>15*1024**3
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())+384*1024**2<24*1024**3
        torch.set_num_threads(4)
        sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'methods/elsc')]
        from elsc.models.retriever import ELSCRetriever
        from elsc.train import _optimizer,_seed_everything,_amp_settings
        from elsc.losses.coarse import balanced_clcl_loss
        from slr_common.upstream.factory import _load_cico_core_from_state,load_cico_tokenizer
        from slr_common.data.cico_dataset import CiCoFeatureDataset
        from slr_common.data.tokenize import CiCoCollator
        from slr_common.data.group_sampler import CaptionGroupSampler
        root=ROOT/f'runs/{cli.dataset}_base_b512_s42'
        configpath=root/'resolved_config.yaml'
        cfg=yaml.safe_load(configpath.read_text())
        assert cfg['method']=='baseline' and not cfg['model']['adapter']['enabled']
        selection=json.loads((root/'selection.json').read_text())
        checkpoint=root/selection['checkpoint']
        assert sha(checkpoint)==selection['checkpoint_sha256']
        _seed_everything(42)
        device=torch.device('cuda')
        raw=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        state={n.removeprefix('core.'):v for n,v in raw['model'].items() if n.startswith('core.')}
        sys.path.insert(0,str(ROOT/cfg['upstream']['cico_root']))
        core=_load_cico_core_from_state(cfg,state,cico_root=ROOT/cfg['upstream']['cico_root'],device=device)
        model=ELSCRetriever(core,input_dim=1024,hidden_dim=cfg['model']['adapter']['hidden_dim'],
                           text_dim=core.clip.text_projection.shape[-1],core_frozen=False,adapter_enabled=False).to(device)
        model.load_state_dict(raw['model'],strict=True)
        optimizer=_optimizer(model,cfg)
        names={id(p):n for n,p in model.named_parameters()}
        assert [n for n,p in model.named_parameters() if p.requires_grad]==raw['trainable_parameter_names']
        for group,stored in zip(optimizer.param_groups,raw['optimizer']['param_groups'],strict=True):
            assert len(group['params'])==len(stored['params'])
            for key in ['betas','eps','weight_decay']:
                assert group[key]==stored[key]
            group['lr']=stored['lr']
        report.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config_sha256=sha(configpath),
                      checkpoint_step=raw['step'],torch=torch.__version__,hardware=torch.cuda.get_device_name(),
                      diagnostic_state='fresh_zero_moments_step1_at_stored_group_LR_NOT_resume')
        report['source_sha256']={str(p):sha(p) for p in [
            ROOT/'methods/elsc/elsc/train.py',ROOT/'methods/elsc/elsc/losses/coarse.py',
            ROOT/'shared/slr_common/upstream/cico_bridge.py',ROOT/'shared/slr_common/upstream/factory.py',
            ROOT/'shared/slr_common/data/cico_dataset.py',ROOT/'shared/slr_common/data/tokenize.py',
            ROOT/'third_party/SLRT/CiCo/CLCL/modules/modeling.py',
            __import__('pathlib').Path(__import__('torch.optim.adam',fromlist=['x']).__file__)]}
        del state,raw
        manifest=ROOT/cfg['data']['train_manifest']
        data=CiCoFeatureDataset(manifest,feature_len=64,alpha=cfg['data']['alpha'],split='train')
        if cfg['train'].get('one_video_per_caption_group',False):
            sampler=CaptionGroupSampler(data.records,seed=42)
            sampler.set_epoch(0)
            indices=list(sampler)[:cli.batch_size]
        else:
            indices=torch.randperm(len(data),generator=torch.Generator().manual_seed(42)).tolist()[:cli.batch_size]
        collator=CiCoCollator(load_cico_tokenizer(cfg),32,augment=True,seed=42)
        batch=collator([data[i] for i in indices])
        features={str(p):sha(p) for i in indices for p in
                  [data.records[i].feature_aware,data.records[i].feature_agnostic]}
        atomic_json(out/'feature_hashes.json',features)
        report.update(train_manifest_sha256=sha(manifest),batch_indices=indices,batch_ids=batch['pair_id'],
                      feature_hashes_sha256=sha(out/'feature_hashes.json'))
        before=tensor_digest(model.state_dict())
        model.train()
        enabled,dtype=_amp_settings(cfg,device)
        assert enabled and dtype==torch.bfloat16
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=dtype):
            video,_=model.encode_video(batch['h'].to(device),batch['valid'].to(device))
            clean=model.encode_text(*(v.to(device) for v in batch['clean_text']))
            aug=model.encode_text(*(v.to(device) for v in batch['aug_text']))
            a,b=model.bridge.score(video,clean,text_aug=aug,objective=True)
            loss=balanced_clcl_loss(a,b,dual_mix=cfg['model']['dual_mix'],mix_design=cfg['model']['mix_design'])
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['train']['grad_clip_norm'])
        assert torch.isfinite(norm)
        gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        assert any(torch.count_nonzero(g)>0 for g in gradients.values())
        torch.save(dict(gradients=gradients,ids=batch['pair_id'],checkpoint_sha256=report['checkpoint_sha256'],
                        loss=float(loss),preclip_norm=float(norm)),out/'gradients.pt')
        report.update(loss=float(loss),preclip_gradient_norm=float(norm),gradient_sha256=sha(out/'gradients.pt'),
                      gradient_tensor_count=len(gradients))
        print(json.dumps({k:report[k] for k in ['loss','preclip_gradient_norm','gradient_tensor_count']}),flush=True)
        del video,clean,aug,a,b,loss,gradients
        rows={}
        sums={}
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is None: continue
                name=names[id(parameter)]
                grad=parameter.grad.detach()
                low,lm,lv=isolated_step(parameter,grad,group)
                mixed,mm,mv=isolated_step(parameter,grad,group,moment_fp32=True)
                high,hm,hv=isolated_step(parameter,grad,group,master_fp32=True)
                foreach,fm,fv=isolated_step(parameter,grad,group,foreach=True)
                assert all(torch.isfinite(x).all() for x in [low,lm,lv,mixed,mm,mv,high,hm,hv,foreach,fm,fv])
                assert torch.equal(mm,hm) and torch.equal(mv,hv)
                if parameter.dtype==torch.float32:
                    assert torch.equal(low,mixed) and torch.equal(low,high)
                native_u=normalized_update(lm,lv,group)
                high_u=normalized_update(hm,hv,group)
                swapped_u=normalized_update(lm,hv,group)
                lowdelta=low.float()-parameter.float()
                highdelta=high-parameter.float()
                row=dict(dtype=str(parameter.dtype),elements=parameter.numel(),
                    native_m_nonzero_v_zero=int(((lm!=0)&(lv==0)).sum()),
                    native_v_zero_reference_positive=int(((lv==0)&(hv>0)).sum()),
                    native_update_sq=squares(native_u),reference_update_sq=squares(high_u),
                    native_vs_reference_update_error_sq=squares(native_u-high_u),
                    denominator_swap_effect_sq=squares(native_u-swapped_u),
                    native_weight_delta_sq=squares(lowdelta),reference_weight_delta_sq=squares(highdelta),
                    moment_only_weight_effect_sq=squares(mixed.float()-low.float()),
                    native_vs_reference_weight_error_sq=squares(low.float()-high),
                    mixed_vs_reference_weight_error_sq=squares(mixed.float()-high),
                    foreach_weight_effect_sq=squares(foreach.float()-low.float()),
                    foreach_weight_maxabs=float((foreach.float()-low.float()).abs().max()),
                    changed_low=int((low!=parameter).sum()),changed_mixed=int((mixed!=parameter).sum()))
                rows[name]=row
                total=sums.setdefault(row['dtype'],dict(tensors=0))
                total['tensors']+=1
                for key,value in row.items():
                    if key=='dtype': continue
                    total[key]=max(total.get(key,0),value) if key.endswith('maxabs') else total.get(key,0)+value
        for total in sums.values():
            total['lost_second_fraction']=total['native_v_zero_reference_positive']/total['elements']
            den=total['native_weight_delta_sq']
            total['moment_only_weight_relative_l2']=math.sqrt(total['moment_only_weight_effect_sq']/den) if den else None
            total['foreach_weight_relative_l2']=math.sqrt(total['foreach_weight_effect_sq']/den) if den else None
            den=total['reference_update_sq']
            total['native_update_relative_error']=math.sqrt(total['native_vs_reference_update_error_sq']/den) if den else None
            total['denominator_swap_relative_effect']=math.sqrt(total['denominator_swap_effect_sq']/den) if den else None
        assert tensor_digest(model.state_dict())==before,'Diagnostic modified original model'
        atomic_json(out/'per_tensor_arithmetic.json',rows)
        half=sums.get('torch.float16',{})
        gate=half.get('lost_second_fraction',0)>.01 and (half.get('moment_only_weight_relative_l2') or 0)>.01
        report.update(status='completed',exit_status=0,model_unchanged=True,sums=sums,activation_gate=gate,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='Real-gradient arithmetic only; no retrieval/confirmation/novel optimizer claim.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-start
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['batch_ids','batch_indices']},indent=2),flush=True)


if __name__=='__main__': main()
