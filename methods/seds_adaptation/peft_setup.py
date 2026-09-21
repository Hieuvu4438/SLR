"""Compose C04 trainable modules without changing native retrieval inference."""
from .low_rank_attention import attach_visual_lora, attach_text_lora
from .train_policies import training_modes


def configure_lora_fusion(model, rank=8, alpha=8., upper_blocks=1, text_lora=False):
    """Freeze native encoders, train existing fusion plus visual-attention LoRA.

    LoRA is attached after freezing so its FP32 factors remain trainable.
    Return active modules for reapplying train/eval modes after evaluation.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attached = attach_visual_lora(model,rank,alpha,upper_blocks)
    if text_lora:
        attached.extend(attach_text_lora(model,rank,alpha))
    for parameter in model.fusion.parameters():
        parameter.requires_grad_(True)
    active = [model.fusion,*[module for _,module in attached]]
    training_modes(model,active)
    return active


def enable_gcn_adaptation(model):
    """C11 composition: train native GCN weights, preserve eval BN/dropout.

    Do not add embed to active training-mode modules; LoRA/fusion policy keeps
    it in eval while autograd flows through the recomputed pose representations.
    """
    model.signbert.embed.eval()
    for parameter in model.signbert.embed.parameters():
        parameter.requires_grad_(True)


def configure_peft_learning_rates(optimizer, model, lora_lr, fusion_lr, encoder_lr=None):
    """Split existing BertAdam groups while preserving all schedule/decay fields.

    Frozen tensors are omitted. Construct before any optimizer update so no
    moment state is discarded. Does not modify loss or invent an optimizer.
    """
    if optimizer.state:
        raise ValueError('Configure learning rates before optimizer has state')
    if lora_lr <= 0 or fusion_lr <= 0:
        raise ValueError('Positive learning rates required')
    if encoder_lr is not None and encoder_lr <= 0:
        raise ValueError('Positive encoder learning rate required')
    rates = dict(lora=lora_lr, fusion=fusion_lr)
    if encoder_lr is not None:
        rates['encoder'] = encoder_lr
    names = {id(p):name for name,p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        partitions = {kind:[] for kind in rates}
        for p in group['params']:
            if not p.requires_grad:
                continue
            name = names[id(p)]
            if '.parametrizations.' in name:
                partitions['lora'].append(p)
            elif name.startswith('fusion.'):
                partitions['fusion'].append(p)
            elif encoder_lr is not None and name.startswith('signbert.embed.'):
                partitions['encoder'].append(p)
            else:
                raise ValueError(f'Unexpected trainable parameter: {name}')
        for kind,parameters in partitions.items():
            if parameters:
                copied = dict(group)
                copied.update(params=parameters,lr=rates[kind],adaptation_group=kind)
                groups.append(copied)
    if not groups:
        raise ValueError('No trainable PEFT parameters')
    if {g['adaptation_group'] for g in groups} != set(rates):
        raise ValueError('Missing registered PEFT parameter group')
    optimizer.param_groups[:] = groups
    return [dict(adaptation_group=g['adaptation_group'],lr=g['lr'],
                 parameters=sum(p.numel() for p in g['params'])) for g in groups]
