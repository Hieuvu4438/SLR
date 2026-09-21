"""Explicit C09 parameter/LR contracts and compact, base-dependent checkpoints."""
import torch
from methods.seds_adaptation.train_policies import configure_trainable_stage, training_modes


def configure_geometry_fusion(model, fusion_only=False, freeze_fusion=False):
    if fusion_only and freeze_fusion:
        raise ValueError('Cannot freeze both geometry and fusion')
    active=configure_trainable_stage(model,'fusion')
    if freeze_fusion:
        for parameter in model.fusion.parameters():
            parameter.requires_grad_(False)
        active=[]
    if not fusion_only:
        active.append(model.pose3d_branch)
        for parameter in active[-1].parameters():
            parameter.requires_grad_(True)
    training_modes(model,active)
    return active


def geometry_learning_rates(optimizer,model,geometry_lr,fusion_lr,fusion_only=False,freeze_fusion=False):
    if fusion_only and freeze_fusion:
        raise ValueError('No trainable branch')
    if optimizer.state or min(geometry_lr,fusion_lr)<=0:
        raise ValueError('Fresh optimizer and positive learning rates required')
    names={id(p):n for n,p in model.named_parameters()}
    groups=[]
    for group in optimizer.param_groups:
        partitions={'geometry':[],'fusion':[]}
        for p in group['params']:
            if not p.requires_grad:continue
            name=names[id(p)]
            if name.startswith('pose3d_branch.'):kind='geometry'
            elif name.startswith('fusion.'):kind='fusion'
            else:raise ValueError(f'Unexpected trainable parameter: {name}')
            partitions[kind].append(p)
        for kind,parameters in partitions.items():
            if parameters:
                groups.append(dict(group,params=parameters,
                    lr=geometry_lr if kind=='geometry' else fusion_lr,adaptation_group=kind))
    expected={'fusion'} if fusion_only else ({'geometry'} if freeze_fusion else {'geometry','fusion'})
    if {g['adaptation_group'] for g in groups}!=expected:
        raise ValueError('Trainable groups differ from registered geometry/control policy')
    optimizer.param_groups[:]=groups
    return [dict(adaptation_group=g['adaptation_group'],lr=g['lr'],weight_decay=g['weight_decay'],
                 parameters=sum(p.numel() for p in g['params'])) for g in groups]


def adaptation_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if k.startswith(('fusion.','pose3d_branch.'))}


def load_adaptation(model,state):
    expected={k for k in model.state_dict() if k.startswith(('fusion.','pose3d_branch.'))}
    if set(state)!=expected:
        raise ValueError('Incomplete or unexpected adaptation keys')
    loaded=model.load_state_dict(state,strict=False)
    if loaded.unexpected_keys or any(k in expected for k in loaded.missing_keys):
        raise ValueError('Adaptation load mismatch')


def check_adaptation_roundtrip(model):
    state=adaptation_state(model)
    with torch.no_grad():
        model.pose3d_branch.output.weight.add_(1.)
    load_adaptation(model,state)
    if not all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in state.items()):
        raise ValueError('Adaptation roundtrip mismatch')
