"""Train-only, clip-aligned RGB SignRep supervision of the native 2D GCN.

No baseline-score anchoring, teacher-selected positives, or inference stream.
"""
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class SignRepDataset(Dataset):
    def __init__(self, native, native_ids, cache_root):
        self.native, self.root = native, Path(cache_root)
        report = json.loads((self.root/'run.json').read_text())
        if report['status'] != 'completed' or report['completed_clips'] != report['total_clips']:
            raise ValueError('Incomplete SignRep cache')
        plan = json.loads((self.root/'plan.json').read_text())
        self.plan = {x['id']: x for x in plan['items'] if x['split'] == 'train'}
        if len(set(native_ids)) != len(native_ids) or not set(self.plan).issubset(native_ids):
            raise ValueError('Ambiguous or missing TRAIN IDs')
        self.indices = [i for i, vid in enumerate(native_ids) if vid in self.plan]
        self.ids = [native_ids[i] for i in self.indices]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample = self.native[self.indices[index]]
        item = self.plan[self.ids[index]]
        starts = torch.as_tensor(sample['body']['clips_start'])
        mask = torch.as_tensor(sample['body']['pose_mask']).bool()
        n = len(item['clip_starts'])
        if (starts.shape != (64,) or mask.shape != (65,) or mask[0]
                or not np.array_equal(starts[:n], item['clip_starts'])
                or not (starts[n:] == -1).all() or mask[1:n+1].any() or not mask[n+1:].all()):
            raise ValueError('SignRep/native clip mask or order mismatch')
        target = torch.zeros(64, 768)
        with np.load(self.root/'train'/(self.ids[index]+'.npz'), allow_pickle=False) as cache:
            if any(not np.array_equal(cache[k], item[k]) for k in ('windows', 'clip_starts')):
                raise ValueError('Cache differs from registered frame mapping')
            values = cache['features'].astype(np.float32)
            if values.shape != (n,768) or not np.isfinite(values).all() or np.any(np.linalg.norm(values,axis=1) == 0):
                raise ValueError('Invalid teacher features')
            target[:n] = torch.from_numpy(values)
        sample['signrep_target'] = target
        return sample


def signrep_collate(native_collate, samples):
    batch = native_collate(samples)
    batch['signrep_target'] = torch.stack([x['signrep_target'] for x in samples])
    return batch


def relational_clip_loss(student, teacher, valid):
    """RKD-D (Park et al., CVPR2019), adapted to valid clips within each video.

    Independently implemented from paper Eqs5-8, not copied donor source.
    No inter-video labels/ranks, temporal-order target, or inference correction.
    """
    pair = valid[:, :, None] & valid[:, None, :]
    pair = pair & ~torch.eye(valid.shape[1],device=valid.device,dtype=torch.bool)[None]
    count = pair.sum((1,2)).clamp_min(1)
    def distances(values):
        x = values.float().masked_fill(~valid[...,None],0)
        center = x.sum(1,keepdim=True)/valid.sum(1).clamp_min(1)[:,None,None]
        x = (x-center).masked_fill(~valid[...,None],0)
        norm = x.square().sum(-1)
        d = (norm[:,:,None]+norm[:,None,:]-2*torch.bmm(x,x.transpose(1,2))).clamp_min(1e-12).sqrt()
        d = d.masked_fill(~pair,0)
        scale = (d.sum((1,2))/count).clamp_min(1e-6)
        return d/scale[:,None,None]
    d_student = distances(student)
    d_teacher = distances(teacher.detach())
    losses = F.smooth_l1_loss(d_student,d_teacher,reduction='none').masked_fill(~pair,0)
    eligible = valid.sum(1)>1
    return (losses.sum((1,2))/count).sum()/eligible.sum().clamp_min(1)


class SignRepTransfer(nn.Module):
    def __init__(self, loss_kind='pointwise'):
        super().__init__()
        if loss_kind not in ('pointwise','relational'):
            raise ValueError(loss_kind)
        self.loss_kind = loss_kind
        self.project = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536,768,bias=False))
        self.loss = None
        self.stats = {}

    def supervision(self, clips, target, valid):
        if clips.shape != (*valid.shape,1536) or target.shape != (*valid.shape,768):
            raise ValueError('Clip target shape mismatch')
        # Separate identity edge observes auxiliary-only encoder gradient.
        edge = clips * 1.
        self.stats = {'valid_teacher_clips': int(valid.sum())}
        if edge.requires_grad:
            def capture(gradient):
                self.stats['auxiliary_gradient_norm'] = float(gradient.float().norm())
                self.stats['auxiliary_gradient_finite'] = bool(torch.isfinite(gradient).all())
            edge.register_hook(capture)
        pred = self.project(edge).float()
        target = target.detach().float()
        if self.loss_kind == 'relational':
            self.loss = relational_clip_loss(pred,target,valid)
            self.stats['valid_relation_pairs'] = int((valid.sum(1)*(valid.sum(1)-1)).sum())
        else:
            self.loss = (1-F.cosine_similarity(pred[valid], target[valid], dim=-1)).mean()
        return self.loss


def attach_signrep_transfer(model, control=False, loss_kind='pointwise'):
    proto = next(model.signbert.parameters())
    module = SignRepTransfer(loss_kind).to(proto.device, proto.dtype)
    model.add_module('signrep_transfer', module)
    original_sign, original_forward = model.get_sign_output, model.forward

    def sign_output(self, right, left, body):
        rgb, pose, mask = original_sign(right, left, body)
        if self.training and not control:
            module.supervision(pose.squeeze(-1).transpose(1,2), body['signrep_target'], ~mask[:,1:].bool())
        return rgb, pose, mask

    def forward(self, *args, **kwargs):
        module.loss = None
        result = original_forward(*args, **kwargs)
        if self.training and not control:
            if module.loss is None or len(result) != 7:
                raise ValueError('Missing auxiliary loss in native forward')
            return (*result, module.loss)
        return result

    model.get_sign_output = MethodType(sign_output, model)
    model.forward = MethodType(forward, model)
    return module


def transfer_learning_rates(optimizer, model, control):
    names = {id(p): n for n,p in model.named_parameters()}
    rates = {'encoder':1e-6, 'fusion':1e-5, 'transfer':1e-4}
    groups = []
    for group in optimizer.param_groups:
        partitions = {k:[] for k in rates}
        for p in group['params']:
            if not p.requires_grad:
                continue
            n = names[id(p)]
            kind = ('encoder' if n.startswith('signbert.embed.') else 'fusion'
                    if n.startswith('fusion.') else 'transfer' if n.startswith('signrep_transfer.') else None)
            if kind is None:
                raise ValueError(n)
            partitions[kind].append(p)
        groups.extend(dict(group,params=ps,lr=rates[k],adaptation_group=k) for k,ps in partitions.items() if ps)
    assert {g['adaptation_group'] for g in groups} == ({'encoder','fusion'} if control else set(rates))
    optimizer.param_groups[:] = groups
    return [dict(adaptation_group=g['adaptation_group'],lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups]


def transfer_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if k.startswith(('fusion.','signbert.embed.','signrep_transfer.'))}


def load_transfer_state(model, state):
    if set(state) != set(transfer_state(model)):
        raise ValueError('Incomplete transfer state; load native release first')
    result = model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and not set(result.missing_keys)&set(state)
