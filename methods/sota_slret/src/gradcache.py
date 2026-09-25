"""Exact single-GPU emulation of upstream SEDS 8-GPU DDP training with a global contrastive batch.

Upstream: 8 ranks x 16 samples. Per rank, the encoders + fusion run on the local 16 samples
(BatchNorm statistics are per-rank), the outputs are all-gathered, the full-batch loss is
computed on every rank, AllGather.backward routes only the local slice's gradient, and DDP
averages gradients over ranks => grad = (1/world) * dL/dtheta. Rank 0's BN running stats
are the ones saved.

Here: (1) encode each chunk without grad (saving RNG state; BN running stats frozen);
(2) compute the full-batch loss on detached chunk outputs with the untouched upstream
similarity code, get d loss / d outputs; (3) re-encode each chunk with grad under the same
RNG state and backprop the cached output-gradients (BN running stats updated only on
chunk 0). Loss is scaled by 1/n_chunks to match DDP gradient averaging.
"""
import contextlib

import numpy as np
import torch
from torch import nn

REP_KEYS = ("seq", "seq_aug", "vis_pose", "vis_rgb", "vis_fusion")
MASK_KEYS = ("text_mask", "text_mask_aug", "video_mask")


def _bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@contextlib.contextmanager
def frozen_bn_stats(model, freeze=True):
    if not freeze:
        yield
        return
    bns = _bn_modules(model)
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0  # running = (1-0)*running + 0*batch  => unchanged
    try:
        yield
    finally:
        for m, mom in zip(bns, saved):
            m.momentum = mom


class _Precomputed(nn.Module):
    """Stands in for model.fusion inside upstream flip_similarity_softmax."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, *_):
        return self.value


def encode_chunk(model, s):
    """Per-rank computation of upstream forward up to (and including) the fusion module."""
    ids, seg, mask = s['pairs_text'], s['pairs_segment'], s['pairs_mask']
    ids = ids.view(-1, ids.shape[-1]); seg = seg.view(-1, seg.shape[-1]); mask = mask.view(-1, mask.shape[-1])
    ids_aug = s['pairs_text_aug'].view(-1, ids.shape[-1])
    mask_aug = s['pairs_mask_aug'].view(-1, mask.shape[-1])
    text_mask, seq = model.get_sequence_output(ids, seg, mask, shaped=False)
    text_mask_aug, seq_aug = model.get_sequence_output(ids_aug, seg, mask_aug, shaped=False)
    body = {'pose': s['body_pose'], 'clips_start': s['body_clips_start'], 'mask': s['body_mask'], 'rgb': s['RGB_feature']}
    video_mask, vp, vr = model.get_visual_output({'pose': s['right_pose']}, {'pose': s['left_pose']}, body, shaped=True)
    vf = model.fusion(vp, vr, video_mask)
    return {"seq": seq, "seq_aug": seq_aug, "vis_pose": vp, "vis_rgb": vr, "vis_fusion": vf,
            "text_mask": text_mask, "text_mask_aug": text_mask_aug, "video_mask": video_mask}


def upstream_loss(model, r):
    """Full-batch loss exactly as upstream CLIP4Clip.forward (training, Filip branch)."""
    saved_fusion, saved_dist = model.fusion, model.distributed
    model.fusion, model.distributed = _Precomputed(r["vis_fusion"]), False
    try:
        (I2T_f, T2I_f, I2T_p, T2I_p, I2T_r, T2I_r, kl_p, kl_r, P2R, R2P) = model.get_similarity_logits(
            r["seq"], r["vis_pose"], r["vis_rgb"], r["text_mask"], r["video_mask"], shaped=True,
            loose_type=model.loose_type, sequence_hidden_aug=r["seq_aug"], text_mask_aug=r["text_mask_aug"])
    finally:
        model.fusion, model.distributed = saved_fusion, saved_dist
    ce, dm = model.loss_fct, model.dual_mix
    # expose detached fusion scores (eval-style mix) for methods that sample hard negatives
    model._last_fusion_sim = (dm * I2T_f + (1 - dm) * T2I_f).detach()

    def pair(i2t, t2i):
        a = ce(i2t) * dm + ce(i2t.T) * (1 - dm)
        b = ce(t2i.T) * dm + ce(t2i) * (1 - dm)
        return (a + b) / 2

    sim_loss, sim_pose, sim_rgb = pair(I2T_f, T2I_f), pair(I2T_p, T2I_p), pair(I2T_r, T2I_r)
    r2p = pair(P2R, R2P) * model.rgb_pose_match_loss if model.rgb_pose_match else torch.zeros_like(sim_loss)
    assert not model.rgb_pose_kl, "KL branch not supported in gradcache path"
    total = sim_loss if model.freeze_exfusion else sim_loss + sim_pose + sim_rgb + r2p
    return total, {"loss": total, "fusion": sim_loss, "pose": sim_pose, "rgb": sim_rgb, "r2p": r2p}


def _rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def _set_rng(st):
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1])


def chunk_bounds(n, n_ranks):
    """Split a batch of n over n_ranks like DDP ranks (near-equal, as DistributedSampler spreads the
    final partial batch); never produces a size-1 chunk (upstream fusion breaks on batch size 1)."""
    sizes = [len(c) for c in np.array_split(np.arange(n), min(n_ranks, max(1, n // 2)))]
    bounds, s = [], 0
    for sz in sizes:
        bounds.append((s, s + sz)); s += sz
    assert all(e - s >= 2 for s, e in bounds) or n < 2
    return bounds


def gradcache_step(model, batch, chunk, loss_fn=upstream_loss, extra_loss=None, n_ranks=None):
    """One training step (forward+backward, no optimizer step). Returns scalar logs.
    `chunk` = per-rank batch of the emulated DDP run (16); n_ranks = full_batch / chunk (8).
    loss_fn(model, reps) -> (loss, logs); extra_loss(model, reps, batch) optional hook."""
    n = batch['pairs_text'].shape[0]
    bounds = chunk_bounds(n, n_ranks if n_ranks else -(-n // chunk))
    sub = lambda s, e: {k: v[s:e] for k, v in batch.items()}
    states, cached = [], []
    with torch.no_grad(), frozen_bn_stats(model, True):
        for s, e in bounds:
            states.append(_rng_state())
            cached.append(encode_chunk(model, sub(s, e)))
    after_state = _rng_state()
    reps = {k: torch.cat([c[k] for c in cached], 0) for k in REP_KEYS + MASK_KEYS}
    leaves = {k: reps[k].detach().requires_grad_(True) for k in REP_KEYS}
    full = dict(reps); full.update(leaves)
    loss, logs = loss_fn(model, full)
    if extra_loss is not None:
        l2, logs2 = extra_loss(model, full, batch)
        loss = loss + l2; logs.update(logs2); logs["loss"] = loss
    assert all(p.grad is None for p in model.parameters()), "call zero_grad(set_to_none=True) first"
    (loss / len(bounds)).backward()
    # Params used directly in the loss (e.g. logit_scale) are replicated per rank upstream and
    # DDP-averaged => full dL/dp, not (1/world) dL/dp. Undo the 1/n scaling for them.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(len(bounds))
    grads = {k: leaves[k].grad for k in REP_KEYS}
    for ci, (s, e) in enumerate(bounds):
        _set_rng(states[ci])
        with frozen_bn_stats(model, freeze=(ci != 0)):
            out = encode_chunk(model, sub(s, e))
        tensors = [out[k] for k in REP_KEYS if grads[k] is not None]
        gs = [grads[k][s:e] for k in REP_KEYS if grads[k] is not None]
        torch.autograd.backward(tensors, gs)
    _set_rng(after_state)
    return {k: float(v) for k, v in logs.items()}
