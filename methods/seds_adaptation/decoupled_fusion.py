"""C17: DCL on native fused retrieval logits only; no teacher or inference change.

Independent implementation of Yeh et al., ECCV2022 / arXiv:2110.06848 Eq5.
Adaptation uses existing cross-modal negatives, not SimCLR's two-view pool.
"""
import json
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.data import Subset


def decoupled_cross_entropy(logits):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or len(logits) < 2:
        raise ValueError('DCL requires square logits and at least two paired samples')
    x = logits.float()
    diagonal = torch.eye(len(x),device=x.device,dtype=torch.bool)
    return (torch.logsumexp(x.masked_fill(diagonal,-torch.inf),dim=1)-x.diag()).mean()


def paired_loss(i2t,t2i,dual_mix,criterion):
    a = criterion(i2t)*dual_mix + criterion(i2t.T)*(1-dual_mix)
    b = criterion(t2i.T)*dual_mix + criterion(t2i)*(1-dual_mix)
    return (a+b)/2


def native_cross_entropy(logits):
    return -F.log_softmax(logits,dim=-1).diag().mean()


def blend_fused_loss(native,decoupled,weight):
    if not 0 <= weight <= 1:
        raise ValueError('DCL weight must be finite and between zero and one')
    if weight == 1:
        return decoupled
    if weight == 0:
        return native
    return (1-weight)*native+weight*decoupled


def registered_native_subset(native,native_ids,reference_path):
    """Read only the historical TRAIN ID list, never SignRep feature files."""
    reference = json.loads(Path(reference_path).read_text())
    if reference.get('status') != 'completed':
        raise ValueError('Subset reference not completed')
    selected = reference['signrep_data']['train_ids']
    if len(selected) != 512 or len(set(selected)) != 512:
        raise ValueError('Expected 512 unique registered TRAIN IDs')
    lookup = {vid:i for i,vid in enumerate(native_ids)}
    if len(lookup) != len(native_ids) or not set(selected).issubset(lookup):
        raise ValueError('Subset IDs absent or duplicated in native TRAIN')
    return Subset(native,[lookup[vid] for vid in selected]),list(selected)


def attach_decoupled_fusion(model,weight=1.):
    if not 0 <= weight <= 1:
        raise ValueError('DCL weight must be finite and between zero and one')
    if getattr(model,'dcl_attached',False):
        raise ValueError('DCL already attached')
    if model.freeze_exfusion or model.rgb_pose_kl:
        raise ValueError('C17 requires the registered native seven-loss recipe')
    original_forward = model.forward
    original_similarity = model.get_similarity_logits
    model.dcl_attached = True
    model.dcl_stats = {}
    model._dcl_pending = None
    model._dcl_in_forward = False

    def similarity(this,*args,**kwargs):
        values = original_similarity(*args,**kwargs)
        if this.training and this._dcl_in_forward:
            if this._dcl_pending is not None:
                raise RuntimeError('Unexpected multiple similarity calls')
            this._dcl_pending = values[:2]
        return values

    def forward(this,*args,**kwargs):
        if not this.training:
            return original_forward(*args,**kwargs)
        if this._dcl_in_forward:
            raise RuntimeError('Reentrant DCL forward is unsupported')
        this._dcl_in_forward = True
        this._dcl_pending = None
        try:
            native = original_forward(*args,**kwargs)
            if len(native) != 7 or this._dcl_pending is None:
                raise RuntimeError('Native fused logits/seven-loss contract changed')
            i2t,t2i = this._dcl_pending
            decoupled = paired_loss(i2t,t2i,this.dual_mix,decoupled_cross_entropy)
            replacement = blend_fused_loss(native[1],decoupled,weight)
            if not replacement.requires_grad:
                raise RuntimeError('DCL loss detached from the trainable model')
            with torch.no_grad():
                reproduced = paired_loss(i2t,t2i,this.dual_mix,native_cross_entropy)
                if not torch.allclose(reproduced,native[1],rtol=1e-5,atol=1e-6):
                    raise RuntimeError('Native fused CE reconstruction mismatch')
                q = torch.cat([-torch.expm1(F.log_softmax(x.float(),dim=-1).diag())
                               for x in (i2t,i2t.T,t2i.T,t2i)])
                this.dcl_stats = dict(fusion_native_ce=float(native[1]),
                    fusion_decoupled_loss=float(decoupled),fusion_effective_loss=float(replacement),
                    dcl_weight=weight,effective_q_mean=float((weight+(1-weight)*q).mean()),
                    npc_q_mean=float(q.mean()),
                    npc_q_p10=float(q.quantile(.1)),npc_q_median=float(q.median()),
                    npc_q_p90=float(q.quantile(.9)),native_fusion_reconstruction_passed=True)
            # Preserve native branch/KL/matching losses exactly. Do not cancel
            # native total algebraically: its saturated fused CE is unused.
            return (replacement+sum(native[2:]),replacement,*native[2:])
        finally:
            this._dcl_pending = None
            this._dcl_in_forward = False

    model.get_similarity_logits = MethodType(similarity,model)
    model.forward = MethodType(forward,model)
