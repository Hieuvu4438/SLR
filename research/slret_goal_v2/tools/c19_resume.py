"""Strict recovery of the interrupted C19 pilot, preserving FP32 BertAdam state."""
import copy
import hashlib
import json
from pathlib import Path
import torch

PARENT = 'seds-global-exchange-cross-001'
LAST_SHA = 'f62c84e34d2568beafcde31a22a35bbdf9ae5be55084aadb3c91a2fbf96fb81f'
GATES = ('global_exchange_initial_identity_passed', 'global_exchange_outputs_updated',
         'global_exchange_all_gradients_passed', 'global_exchange_trained_delta_roundtrip_passed',
         'masked_update_checks_passed', 'delta_roundtrip_passed', 'returned_loss_components_consistent')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def check_config(saved, current):
    exempt = {'run_id','resume_from'}
    if {k:v for k,v in saved.items() if k not in exempt} != {k:v for k,v in current.items() if k not in exempt}:
        raise ValueError('Recovery must not change the scientific recipe')


def read_recovery(root, path, config):
    path = Path(path).resolve()
    expected = root/'artifacts/slret_goal_v2'/PARENT/'last.pt'
    if path != expected or digest(path) != LAST_SHA:
        raise ValueError('Unregistered recovery checkpoint or changed hash')
    source = json.loads((path.parent/'run.json').read_text())
    checkpoint = torch.load(path,map_location='cpu')
    check_config(checkpoint['adaptation'],config)
    if checkpoint['next_batch_index'] != 112 or len(checkpoint['batches']) != 160:
        raise ValueError('Expected saved update112 of160')
    if source['candidate'] != 'C19_global_bottleneck_cross' or source['test_loaded']:
        raise ValueError('Wrong parent experiment')
    if not all(source.get(k) for k in GATES):
        raise ValueError('Parent integration gates incomplete')
    if checkpoint['optimizer_precision'] != 'fp32_moments_native_parameters' or checkpoint['scheduler'] is not None:
        raise ValueError('Unsupported optimizer/scheduler state')
    changed = []
    for relative, old in checkpoint['code_sha256'].items():
        snapshot = path.parent/'source'/relative
        if digest(snapshot) != old:
            raise ValueError('Parent source snapshot changed: '+relative)
        if digest(root/relative) != old:
            changed.append(relative)
    if changed != ['research/slret_goal_v2/tools/train_seds_extended.py']:
        raise ValueError('Only reviewed trainer recovery wiring may differ: '+str(changed))
    if checkpoint['base_checkpoint']['sha256'] != digest(checkpoint['base_checkpoint']['path']):
        raise ValueError('Native release checkpoint changed')
    selection = json.loads((path.parent/'selection.json').read_text())
    if selection['step'] > 112 or digest(selection['checkpoint']) != selection['checkpoint_sha256']:
        raise ValueError('Selection is not valid for saved state')
    rows = [json.loads(line) for line in (path.parent/'train_steps.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != list(range(1,122)):
        raise ValueError('Unexpected parent training log')
    return checkpoint, source, selection, rows[:112]


def restore_bertadam(optimizer, saved, step):
    """Avoid Optimizer.load_state_dict casting FP32 moments to FP16 parameters."""
    if optimizer.state or len(optimizer.param_groups) != len(saved['param_groups']):
        raise ValueError('Fresh matching optimizer groups required')
    pending = {}
    seen = set()
    for live, old in zip(optimizer.param_groups,saved['param_groups']):
        if {k:v for k,v in live.items() if k!='params'} != {k:v for k,v in old.items() if k!='params'}:
            raise ValueError('Optimizer schedule/group metadata mismatch')
        if len(live['params']) != len(old['params']):
            raise ValueError('Optimizer parameter count mismatch')
        for parameter, index in zip(live['params'],old['params']):
            if index in seen:
                raise ValueError('Duplicate optimizer parameter index')
            seen.add(index)
            state = saved['state'][index]
            if set(state) != {'step','next_m','next_v'} or state['step'] != step:
                raise ValueError('Incomplete or wrong optimizer step')
            restored = {'step':step}
            for key in ('next_m','next_v'):
                value = state[key]
                if value.dtype != torch.float32 or value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid FP32 moment: '+key)
                restored[key] = value.to(device=parameter.device,copy=True)
            pending[parameter] = restored
    if seen != set(saved['state']):
        raise ValueError('Unused optimizer state')
    optimizer.state.update(pending)


def inherit_progress(report, parent, start):
    for key in GATES:
        report[key] = parent[key]
    report['criterion'] = parent['criterion']
    report['evaluations'] = {k:copy.deepcopy(v) for k,v in parent['evaluations'].items() if int(k)<=start}
    if set(report['evaluations']) != {'0','80'}:
        raise ValueError('Expected completed parent DEV0/80')
    report['steps'] = start
    report['step0'] = 'inherited parent initial DEV and identity; resumed learned weights, no replay'
    report['resume'] = dict(parent_run=PARENT,checkpoint_sha256=LAST_SHA,
        next_batch_index=start,parent_recorded_steps=parent['steps'],
        discarded_unsaved_updates=parent['steps']-start,
        inherited_gate_source='parent run.json; original source snapshots verified',
        inherited_evaluation_source='parent run.json DEV0/80; selected checkpoint retained')
