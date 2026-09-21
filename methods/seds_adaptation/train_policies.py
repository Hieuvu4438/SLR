"""Selective existing-module adaptation (C02), no new inference component."""


def configure_trainable_stage(model, stage, upper_blocks=1):
    """Configure before constructing the stage's optimizer/wrapper.

    Lower representation modules stay in eval mode so freezing weights does not
    silently update BatchNorm statistics. Reapply training_modes after eval.
    This helper does not change tensors, reset optimizers, or alter gradients.
    """
    if stage not in ('fusion','upper') or upper_blocks < 1:
        raise ValueError('Invalid adaptation stage')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    active = [model.fusion]
    if stage == 'upper':
        for clip in (model.clip,model.clip_rgb):
            blocks = clip.visual.transformer.resblocks
            if upper_blocks > len(blocks):
                raise ValueError('Requested more upper visual blocks than exist')
            active.extend(list(blocks)[-upper_blocks:])
            # Scope this policy to transformer blocks. encode_image applies
            # ln_post/proj afterward; those used parameters stay frozen by design.
    for module in active:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    training_modes(model,active)
    return active


def training_modes(model, active_modules):
    model.eval()
    # Native SEDS returns losses only when its top-level training flag is true.
    # Set this flag directly, without recursively re-enabling frozen BN/dropout.
    model.training = True
    for module in active_modules:
        module.train()


def configure_interaction_fusion(model):
    """Train only C03 interaction and fusion; backpropagate through frozen tail."""
    active = configure_trainable_stage(model, 'fusion')
    interaction = model.signbert.articulator_interaction
    for parameter in interaction.parameters():
        parameter.requires_grad_(True)
    active.append(interaction)
    training_modes(model, active)
    return active
