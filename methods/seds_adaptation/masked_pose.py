"""C08: train-only masked coordinate reconstruction, borrowed objective.

Native crop coordinates are pixels on a 256x256 crop, NOT normalized XY.
The auxiliary target is (XY / 256 - .5); inference uses the untouched path.
"""
from types import MethodType
import torch
from torch import nn
from methods.seds_adaptation.train_policies import configure_trainable_stage


class MaskedPose(nn.Module):
    def __init__(self, width=1536, ratio=.2, seed=42):
        super().__init__()
        if not 0 < ratio < 1:
            raise ValueError('Mask ratio must be in (0,1)')
        self.decoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128),
                                     nn.GELU(), nn.Linear(128, 98))
        self.ratio, self.seed = ratio, seed
        self.generator = None
        self.loss = None
        self.stats = {}

    def corrupt(self, pose, covered):
        target = torch.cat([pose[k] for k in ('left', 'right', 'body')], dim=2).detach()
        if target.shape[-2:] != (49, 2) or covered.shape != target.shape[:2]:
            raise ValueError('Expected 49 XY joints and exact frame coverage')
        # Native collator zero pads. Confidence was discarded upstream: do not
        # manufacture it. Exclude zero XY pairs conservatively (including borders).
        valid = covered[..., None] & torch.isfinite(target).all(-1) & target.ne(0).any(-1)
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite native pose')
        if self.generator is None:
            self.generator = torch.Generator(device=target.device).manual_seed(self.seed)
        selected = (torch.rand(valid.shape, device=target.device, generator=self.generator) < self.ratio) & valid
        corrupted = target.masked_fill(selected[..., None], 0.)
        result = dict(pose)
        for key, part in zip(('left', 'right', 'body'), corrupted.split((21, 21, 7), dim=2)):
            result[key] = part
        self.stats = dict(masked_joints=int(selected.sum()), valid_joints=int(valid.sum()),
                          masked_fraction=float(selected.sum()/valid.sum().clamp_min(1)))
        return result, target / 256. - .5, selected

    def reconstruct(self, features, target, selected):
        # A separate identity edge lets the hook observe auxiliary-only gradient
        # without a second autograd traversal (unsafe with DDP's reducer).
        auxiliary_features = features * 1.
        if auxiliary_features.requires_grad:
            def capture_gradient(gradient):
                self.stats['auxiliary_feature_gradient_finite'] = bool(torch.isfinite(gradient).all())
                self.stats['auxiliary_feature_gradient_norm'] = float(gradient.detach().float().norm())
            auxiliary_features.register_hook(capture_gradient)
        prediction = self.decoder(auxiliary_features).reshape(*target.shape)
        error = (prediction.float() - target.float()).square().mean(-1)
        self.loss = (error * selected).sum() / selected.sum().clamp_min(1)
        return self.loss


def attach_masked_pose(model, control=False, ratio=.2, seed=42):
    prototype = next(model.signbert.parameters())
    module = MaskedPose(ratio=ratio, seed=seed).to(prototype.device, prototype.dtype)
    module.train(model.training)
    model.add_module('masked_pose', module)
    original_sign = model.get_sign_output
    original_gcn = model.signbert.gcn_emb
    original_forward = model.forward
    coverage = None

    def sign_output(self, right, left, body):
        nonlocal coverage
        module.loss = None
        if module.training and not control:
            b, t = body['pose'].shape[:2]
            coverage = torch.zeros((b, t), dtype=torch.bool, device=body['pose'].device)
            for i, starts in enumerate(body['clips_start']):
                for start in starts.tolist():
                    if start >= 0:
                        coverage[i, start:start+self.task_config.slide_windows] = True
        try:
            return original_sign(right, left, body)
        finally:
            coverage = None

    def gcn_output(self, pose):
        if not module.training or control:
            return original_gcn(pose)
        if coverage is None:
            raise ValueError('Masked training requires native clip coverage')
        corrupted, target, selected = module.corrupt(pose, coverage)
        encoded = original_gcn(corrupted)
        module.reconstruct(encoded['feat'], target, selected)
        return encoded

    def forward_with_auxiliary(self, *args, **kwargs):
        result = original_forward(*args, **kwargs)
        if self.training and not control:
            if module.loss is None or not isinstance(result, tuple) or len(result) != 7:
                raise ValueError('Expected native seven losses and current reconstruction')
            # DDP find_unused_parameters traverses returned tensors. Returning
            # this branch prevents it marking decoder parameters unused early.
            auxiliary = module.loss
            module.loss = None
            return (*result, auxiliary)
        return result

    model.get_sign_output = MethodType(sign_output, model)
    model.signbert.gcn_emb = MethodType(gcn_output, model.signbert)
    model.forward = MethodType(forward_with_auxiliary, model)
    return module


def configure_masked_pose(model, control=False, fusion_only=False, clip_temporal=False):
    if fusion_only and not control:
        raise ValueError('Fusion-only ablation requires clean control')
    if clip_temporal and (not control or fusion_only):
        raise ValueError('Clip-temporal adaptation requires clean GCN control')
    active = configure_trainable_stage(model, 'fusion')
    # Adapt existing GCN weights but keep its dropout/BN in eval mode, in BOTH
    # arms. requires_grad and module.training are deliberately independent.
    if not fusion_only:
        for parameter in model.signbert.embed.parameters():
            parameter.requires_grad_(True)
    if clip_temporal:
        model.signbert.GCN_Conv.eval()
        for parameter in model.signbert.GCN_Conv.parameters():
            parameter.requires_grad_(True)
    if not control:
        active.append(model.masked_pose)
        for parameter in model.masked_pose.parameters():
            parameter.requires_grad_(True)
    return active


def masked_learning_rates(optimizer, model, control=False, fusion_only=False, clip_temporal=False,
                          adaptive_graph=False, graph_lr=1e-4, bone_features=False,
                          bone_lr=1e-5, joint_bilinear=False, temporal_modulation=False,
                          temporal_modulation_lr=1e-4):
    if fusion_only and not control:
        raise ValueError('Fusion-only ablation requires clean control')
    if clip_temporal and (not control or fusion_only):
        raise ValueError('Clip-temporal adaptation requires clean GCN control')
    if optimizer.state:
        raise ValueError('Fresh optimizer required')
    if adaptive_graph and (not control or fusion_only or clip_temporal):
        raise ValueError('Adaptive graph requires clean GCN control recipe')
    if bone_features and (not control or fusion_only or clip_temporal or adaptive_graph):
        raise ValueError('Bone features require standalone clean GCN control recipe')
    if joint_bilinear and (not control or fusion_only or clip_temporal or adaptive_graph or bone_features):
        raise ValueError('Joint bilinear requires standalone clean GCN control recipe')
    names = {id(p): n for n, p in model.named_parameters()}
    rates = {'encoder': 1e-6, 'fusion': 1e-5, 'decoder': 1e-4}
    if clip_temporal:
        rates['clip_temporal'] = 1e-6
    if adaptive_graph:
        if graph_lr not in (1e-4, 1e-5):
            raise ValueError('Unregistered graph learning rate')
        rates['graph'] = graph_lr
    if bone_features:
        if bone_lr not in (1e-5,1e-4):
            raise ValueError('Unregistered bone learning rate')
        rates['bone'] = bone_lr
    if joint_bilinear:
        rates['bilinear'] = 1e-4
    if temporal_modulation:
        if temporal_modulation_lr != 1e-4:
            raise ValueError('Unregistered temporal modulation learning rate')
        rates['temporal_modulation'] = temporal_modulation_lr
    groups = []
    for group in optimizer.param_groups:
        partitions = {k: [] for k in rates}
        for p in group['params']:
            if not p.requires_grad:
                continue
            name = names[id(p)]
            kind = ('temporal_modulation' if temporal_modulation and name.startswith('temporal_modulation.') else
                    'bilinear' if joint_bilinear and '.joint_bilinear.' in name else
                    'bone' if bone_features and '.bone_features.' in name else
                    'graph' if adaptive_graph and name.endswith('.graph_delta') else
                    'encoder' if name.startswith('signbert.embed.') else
                    'clip_temporal' if clip_temporal and name.startswith('signbert.GCN_Conv.') else
                    'fusion' if name.startswith('fusion.') else
                    'decoder' if name.startswith('masked_pose.') else None)
            if kind is None:
                raise ValueError(f'Unexpected trainable tensor {name}')
            partitions[kind].append(p)
        for kind, params in partitions.items():
            if params:
                groups.append(dict(group, params=params, lr=rates[kind], adaptation_group=kind))
    expected = {'fusion'} if fusion_only else ({'encoder', 'fusion'} if control else set(rates))
    if clip_temporal:
        expected.add('clip_temporal')
    if adaptive_graph:
        expected.add('graph')
    if bone_features:
        expected.add('bone')
    if joint_bilinear:
        expected.add('bilinear')
    if temporal_modulation:
        expected.add('temporal_modulation')
    if {g['adaptation_group'] for g in groups} != expected:
        raise ValueError('Missing adaptation groups')
    optimizer.param_groups[:] = groups
    return [dict(adaptation_group=g['adaptation_group'], lr=g['lr'],
                 parameters=sum(p.numel() for p in g['params'])) for g in groups]
