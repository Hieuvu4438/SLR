"""C07: anatomical-token exchange BEFORE native GCN part pooling.

Known attention operator; novelty/efficacy are unproven. Unlike C03 this sees
individual joint features before max-pooling. No extra modality, teacher,
confidence gate, score residual, temporal warp or linguistic pseudo-label.
"""
from types import MethodType
import torch
from torch import nn
from methods.seds_adaptation.train_policies import configure_trainable_stage, training_modes


class JointExchange(nn.Module):
    def __init__(self, width=256, rank=32, counts=(21, 21, 7), mode='cross'):
        super().__init__()
        if mode not in ('cross', 'within') or min(counts) < 1:
            raise ValueError('Invalid anatomical exchange configuration')
        self.counts, self.mode = tuple(counts), mode
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.identity = nn.Parameter(torch.empty(sum(counts), rank))
        nn.init.normal_(self.identity, std=.02)
        self.query = nn.Linear(width, rank, bias=False)
        self.key = nn.Linear(width, rank, bias=False)
        self.value = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.output.weight)
        groups = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        allowed = groups[:, None] != groups[None, :] if mode == 'cross' else groups[:, None] == groups[None, :]
        self.register_buffer('allowed', allowed)

    def forward(self, x, valid):
        # Flattened batch/frame axis, actual anatomical joint identities retained.
        if x.ndim != 3 or x.shape[1] != sum(self.counts) or valid.shape != x.shape[:2]:
            raise ValueError('Expected [frames,joints,channels] and joint validity')
        z = self.norm(x)
        q, k = self.query(z) + self.identity, self.key(z) + self.identity
        logits = (q @ k.transpose(-1, -2)) / q.shape[-1]**.5
        permitted = self.allowed[None] & valid[:, None, :].bool()
        # Finite empty rows: do not let an absent hand become an artificial key.
        weights = logits.masked_fill(~permitted, -1e4).softmax(-1) * permitted
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        delta = self.output(weights @ self.value(z)) * valid[..., None]
        return x + delta


def _encode_joints(gcn, pose):
    """Native ST_GCN_Model.forward prefix, unchanged ops with temporal_pad=0."""
    b, t, joints, coords = pose.shape
    x = pose.contiguous().view(b*t, 1, joints, coords).permute(0, 3, 1, 2).unsqueeze(-1)
    n, c, time, v, m = x.shape
    x = x.permute(0, 4, 3, 1, 2).contiguous().view(n*m, v*c, time)
    x = gcn.data_bn(x).view(n, m, v, c, time)
    x = x.permute(0, 1, 3, 4, 2).contiguous().view(n*m, c, 1, -1)
    for block in gcn.st_gcn_networks:
        x, _ = block(x, gcn.A)
    return x.view(n, -1, time, v)


def _finish_joints(gcn, x, batch, sequence):
    """Native part pooling, graph/temporal tail; reuse all pretrained weights."""
    parts = [gcn.graph_max_pool(x[:, :, :, nodes], (1, len(nodes))) for nodes in gcn.graph.part]
    x = torch.cat(parts, -1)
    x, _ = gcn.st_gcn_pool[0](x.view(batch*sequence, -1, 1, len(parts)), gcn.A_pool.clone())
    x, _ = gcn.st_gcn_pool[1](x, gcn.A_pool.clone())
    x = gcn.graph_max_pool(x.view(batch*sequence, -1, 1, len(parts)), (1, len(parts)))
    x = gcn.conv4(x).view(batch, sequence, -1).contiguous().permute(0, 2, 1)
    return gcn.tcn_full_b2(gcn.tcn_full_b1(x)).contiguous().permute(0, 2, 1)


def attach_joint_exchange(signbert, rank=32, mode='cross'):
    if hasattr(signbert, 'joint_exchange'):
        raise ValueError('Joint exchange already attached')
    embed = signbert.embed
    gcns = (embed.st_gcn_hand, embed.st_gcn_hand, embed.st_gcn_body)
    counts = tuple(g.graph.num_node_each for g in gcns)
    prototype = next(signbert.parameters())
    module = JointExchange(gcns[0].inter_channels[-1], rank, counts, mode).to(
        device=prototype.device, dtype=prototype.dtype)
    signbert.add_module('joint_exchange', module)

    def adapted(self, pose):
        # Same left/right/body output order as native, no change to crop frames.
        inputs = [pose[k] for k in ('left', 'right', 'body')]
        b, t = inputs[0].shape[:2]
        nodes = [_encode_joints(g, p) for g, p in zip(gcns, inputs)]
        joint_features = torch.cat(nodes, dim=-1).squeeze(2).transpose(1, 2)
        # Validity excludes zero-padded frames/absent whole parts. Individual
        # zero coordinates are not confidence estimates and are not reweighted.
        valid = torch.cat([p.abs().sum((-1, -2)).ne(0)[..., None].expand(b, t, n)
                           for p, n in zip(inputs, counts)], -1).reshape(b*t, -1)
        exchanged = torch.cat([self.joint_exchange(x, m) for x, m in
                               zip(joint_features.split(128), valid.split(128))], 0)
        parts = exchanged.transpose(1, 2).unsqueeze(2).split(counts, dim=-1)
        pose['feat'] = torch.cat([_finish_joints(g, x, b, t) for g, x in zip(gcns, parts)], -1)
        return pose

    signbert.gcn_emb = MethodType(adapted, signbert)
    return module


def configure_joint_fusion(model):
    active = configure_trainable_stage(model, 'fusion')
    active.append(model.signbert.joint_exchange)
    for p in active[-1].parameters():
        p.requires_grad_(True)
    training_modes(model, active)
    return active


def configure_joint_learning_rates(optimizer, model, joint_lr, fusion_lr):
    """Native non-CLIP/non-sign groups otherwise inherit sign_lr, NOT args.lr.

    Apply before the first update; preserve schedule/decay fields. Only the
    registered joint module and fusion may be trainable in this experiment.
    """
    if optimizer.state or min(joint_lr, fusion_lr) <= 0:
        raise ValueError('Positive LRs and fresh optimizer required')
    names = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        partitions = {'joint': [], 'fusion': []}
        for p in group['params']:
            if not p.requires_grad:
                continue
            name = names[id(p)]
            if name.startswith('signbert.joint_exchange.'):
                kind = 'joint'
            elif name.startswith('fusion.'):
                kind = 'fusion'
            else:
                raise ValueError(f'Unexpected trainable parameter: {name}')
            partitions[kind].append(p)
        for kind, parameters in partitions.items():
            if parameters:
                updated = dict(group)
                updated.update(params=parameters, lr=joint_lr if kind == 'joint' else fusion_lr,
                               adaptation_group=kind)
                groups.append(updated)
    if {g['adaptation_group'] for g in groups} != {'joint', 'fusion'}:
        raise ValueError('Both joint and fusion groups required')
    optimizer.param_groups[:] = groups
    return [{k:g[k] for k in ('adaptation_group','lr','weight_decay')}
            for g in groups]
