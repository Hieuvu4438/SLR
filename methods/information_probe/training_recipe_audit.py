"""Actual upstream freeze AST and optimizer semantics; no retrieval training."""
import ast
import copy
import importlib.util
import json
import os
import time
import traceback
from types import SimpleNamespace

import torch
import yaml

from .clean_initialization_audit import GENERIC_SHA, OUT, initialize
from .clean_train_calibration import lr_factor
from .common import ROOT, dump, sha

MAIN = ROOT/'third_party/SLRT/CiCo/CLCL/main_task_retrieval.py'
OPT = ROOT/'third_party/SLRT/CiCo/CLCL/modules/optimization.py'


def optimizer_module():
    spec = importlib.util.spec_from_file_location('audited_cico_optimization', OPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_source_freeze(model, layer=0):
    tree = ast.parse(MAIN.read_text())
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and "hasattr(model, 'clip')" in ast.unparse(node.test)
                  and 'args.freeze_layer_num > -1' in ast.unparse(node.test)]
    assert len(candidates) == 1
    block = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(candidates[0])], type_ignores=[]))
    exec(compile(block, str(MAIN), 'exec'), {'model': model, 'args': SimpleNamespace(freeze_layer_num=layer, linear_patch='2d')})


def explicit_uncorrected(p, m, v, g, lr, wd=0.):
    m = .9*m+.1*g
    v = .98*v+.02*g.square()
    return p-lr*(m/(v.sqrt()+1e-6)+wd*p), m, v


def synthetic_checks():
    module = optimizer_module()
    p0 = torch.tensor([1., -2., .5], dtype=torch.float64)
    g = torch.tensor([.2, -.3, .1], dtype=torch.float64)
    bp, ap = torch.nn.Parameter(p0.clone()), torch.nn.Parameter(p0.clone())
    bo = module.BertAdam([bp], lr=1e-4, b1=.9, b2=.98, e=1e-6, weight_decay=0., max_grad_norm=-1)
    ao = torch.optim.AdamW([ap], lr=1e-4, betas=(.9,.98), eps=1e-6, weight_decay=0.)
    ref, m, v = p0.clone(), torch.zeros_like(p0), torch.zeros_like(p0)
    constant, maximum = [], 0.
    for step in range(1, 101):
        bp.grad, ap.grad = g.clone(), g.clone()
        bo.step(); ao.step()
        ref, m, v = explicit_uncorrected(ref, m, v, g, 1e-4)
        maximum = max(maximum, float((bp.detach()-ref).abs().max()))
        if step in (1, 2, 10, 100):
            constant.append({'step': step, 'bert_delta': (bp.detach()-p0).tolist(), 'adamw_delta': (ap.detach()-p0).tolist(),
                             'max_parameter_difference': float((bp.detach()-ap.detach()).abs().max())})
    assert maximum < 1e-12, maximum
    # Fixed varying gradients: actual caller-global clip in both arms.
    bp = [torch.nn.Parameter(p0.clone()), torch.nn.Parameter(torch.tensor([3.,-1.], dtype=torch.float64))]
    ap = [torch.nn.Parameter(p.detach().clone()) for p in bp]
    bo = module.BertAdam(bp, lr=1e-4, b1=.9,b2=.98,e=1e-6,weight_decay=.001,
                         max_grad_norm=1., warmup=.1,t_total=1000,schedule='warmup_cosine')
    ao = torch.optim.AdamW(ap,lr=1e-4,betas=(.9,.98),eps=1e-6,weight_decay=.001)
    scheduled = []
    for step in range(1, 1001):
        gradients = [torch.tensor([3., 4., (step%7-3)*.2],dtype=torch.float64),
                     torch.tensor([.1, -.2 if step%2 else .3],dtype=torch.float64)]
        for p, q, grad in zip(bp, ap, gradients):
            p.grad, q.grad = grad.clone(), grad.clone()
        torch.nn.utils.clip_grad_norm_(bp, 1.)
        torch.nn.utils.clip_grad_norm_(ap, 1.)
        before = [p.grad.clone() for p in bp]
        factor = lr_factor(step)
        ao.param_groups[0]['lr'] = 1e-4*factor
        bo.step(); ao.step()
        if step in (1, 2, 10, 100, 500, 1000):
            scheduled.append({'step': step, 'bert_lr_factor': module.warmup_cosine((step-1)/1000,.1),
                'adamw_lr_factor': factor, 'max_parameter_difference': max(float((p-q).detach().abs().max()) for p,q in zip(bp,ap)),
                'extra_internal_clip_max_gradient_difference': max(float((p.grad-prev).abs().max()) for p,prev in zip(bp,before))})
    return {'constant_lr_no_decay_no_clip': constant, 'explicit_bert_formula_max_error': maximum,
            'scheduled_with_decay_global_clip_both': scheduled}


def main():
    path = OUT/'AS-C28-RECIPE_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C28-RECIPE', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C28_protocol.md'),
              'upstream_main_sha256': sha(MAIN), 'upstream_optimizer_sha256': sha(OPT),
              'torch_version': str(torch.__version__), 'method_go': False, 'retrieval_training_updates': 0,
              'dev_or_test_inputs_loaded': False, 'ph_retrieval_release_loaded': False}
    dump(path, result)
    try:
        assert sha(MAIN) == '5f6e3606a170dd5c5a130ddadab67ee1d276f36e619b445f51bf1b6087420418'
        assert sha(OPT) == '7dab5de2bbd1398e07ddbb18667544caae385d8ef9c17c55e0358303ed28b4b8'
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == GENERIC_SHA
        generic = torch.jit.load(str(generic_path),map_location='cpu').state_dict()
        core, init = initialize(config,generic,42)
        del generic
        assert all(p.requires_grad for p in core.parameters())
        apply_source_freeze(core)
        frozen = {n:p for n,p in core.named_parameters() if not p.requires_grad}
        assert set(frozen) == {'clip.token_embedding.weight','clip.positional_embedding'}
        result['freeze_policy'] = {'keys': {n:{'shape':list(p.shape),'numel':p.numel()} for n,p in frozen.items()},
                                   'frozen_parameter_n':sum(p.numel() for p in frozen.values()),
                                   'remaining_trainable_parameter_n':sum(p.numel() for p in core.parameters() if p.requires_grad)}
        mismatch = []
        for name,p in core.named_parameters():
            upstream_decay = not any(x in name for x in ('bias','LayerNorm.bias','LayerNorm.weight'))
            local_decay = p.ndim > 1 and not name.endswith('.bias')
            if upstream_decay != local_decay:
                mismatch.append({'name':name,'numel':p.numel(),'upstream_decay':upstream_decay,
                                 'AS_C20_decay':local_decay,'upstream_trainable':p.requires_grad})
        result['decay_predicate_mismatches'] = mismatch
        run = json.loads((OUT/'AS-C20-TRAIN_run.json').read_text())
        cp = run['checkpoint']['path']
        assert run['status'] == 'completed' and sha(cp) == run['checkpoint']['sha256']
        saved = torch.load(cp, map_location='cpu',weights_only=True)
        result['calibration_checkpoint_sha256'] = run['checkpoint']['sha256']
        drift = {}
        for n,p in frozen.items():
            a, b = p.detach().double(), saved['state_dict'][n].double()
            delta = b-a
            drift[n] = {'relative_l2': float(delta.norm()/a.norm()), 'max_abs':float(delta.abs().max()),
                        'changed_rows':int((delta != 0).any(-1).sum()), 'row_n':p.shape[0]}
        result['AS_C20_table_drift'] = drift
        result['AS_C20_final_logit_scale'] = run['training'][-1]['logit_scale']
        result['upstream_training_loop_logit_scale_cap'] = 100.
        del saved, core
        result['synthetic_optimizer_checks'] = synthetic_checks()
        if time.time()-started > 300:
            raise TimeoutError('AS-C28 timeout300s')
        result.update(status='completed',wall_seconds=time.time()-started)
        dump(path,result)
        print(json.dumps({k:v for k,v in result.items() if k != 'decay_predicate_mismatches'},indent=2))
        print(json.dumps({'decay_predicate_mismatch_tensor_n':len(mismatch),'mismatch_parameter_n':sum(x['numel'] for x in mismatch)}))
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        dump(path,result)
        raise


if __name__ == '__main__':
    main()
