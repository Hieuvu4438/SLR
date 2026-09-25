"""Evaluation that mirrors upstream SEDS `eval_epoch` score construction, returning raw
similarity matrices plus official-compatible and sanity metrics with per-query ranks."""
import numpy as np
import torch

from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from sanity_eval import evaluate as sanity_evaluate, groups_from_cutoffs


@torch.no_grad()
def encode_eval_set(model, dataloader, device):
    """Returns per-batch cached outputs in the same structure upstream eval_epoch builds."""
    ds = dataloader.dataset
    cut = [c - 1 for c in ds.cut_off_points]
    model.eval()
    vids, texts, total = [], [], 0
    for batch in dataloader:
        s = {k: v.to(device) for k, v in batch.items()}
        body = {'pose': s['body_pose'], 'clips_start': s['body_clips_start'], 'mask': s['body_mask'], 'rgb': s['RGB_feature']}
        mask, vp, vr = model.get_visual_output({'pose': s['right_pose']}, {'pose': s['left_pose']}, body, shaped=True, get_hidden=True)
        vids.append((mask, vp, vr))
        b = s['pairs_text'].shape[0]
        keep = [i - total for i in cut if total <= i < total + b]
        if keep:
            tm, so = model.get_sequence_output(s['pairs_text'][keep], s['pairs_segment'][keep], s['pairs_mask'][keep], get_hidden=True)
            texts.append((tm, so))
        total += b
    return vids, texts, ds.cut_off_points


@torch.no_grad()
def score_matrices(model, vids, texts, dual_mix=0.5, streams=("fusion", "pose", "rgb")):
    """Same pairwise loop and 0.5*I2T + 0.5*T2I mixing as upstream _run_on_single_gpu_new_mix.
    Returns dict stream -> [N_video, N_text_groups] float64."""
    out = {k: [] for k in streams}
    idx = {"fusion": (0, 1), "pose": (2, 3), "rgb": (4, 5)}
    for mask, vp, vr in vids:
        row = {k: [] for k in streams}
        for tm, so in texts:
            res = model.get_similarity_logits(so, vp, vr, tm, mask, loose_type=model.loose_type, is_train=True)
            for k in streams:
                i2t, t2i = res[idx[k][0]], res[idx[k][1]]
                row[k].append((i2t * dual_mix + t2i * (1 - dual_mix)).double().cpu().numpy())
        for k in streams:
            out[k].append(np.concatenate(row[k], axis=-1))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def official_metrics(sim_vt, cut_off_points, sim_vt_for_v2t=None):
    """Upstream reshape (pad groups with -inf) + upstream metric functions, verbatim logic."""
    ends = list(cut_off_points)
    starts = [0] + ends[:-1]
    max_len = max(e - s for s, e in zip(starts, ends))
    stacked = np.stack([np.concatenate((sim_vt[s:e], np.full((max_len - e + s, sim_vt.shape[1]), -np.inf)), axis=0)
                        for s, e in zip(starts, ends)], axis=0)
    t2v = compute_metrics(tensor_video_to_text_sim(stacked))
    if sim_vt_for_v2t is not None:  # direction-specific scores (e.g. per-query top-K reranking)
        stacked = np.stack([np.concatenate((sim_vt_for_v2t[s:e], np.full((max_len - e + s, sim_vt.shape[1]), -np.inf)), axis=0)
                            for s, e in zip(starts, ends)], axis=0)
    v2t = tensor_text_to_video_metrics(stacked)
    t2v = {k: v for k, v in t2v.items() if k != "cols"}
    return {"t2v": t2v, "v2t": v2t}


def full_metrics(sim_vt, cut_off_points, sim_vt_for_v2t=None):
    """sim_vt scores T2V (and V2T unless sim_vt_for_v2t is given)."""
    off = official_metrics(sim_vt, cut_off_points, sim_vt_for_v2t)
    vg, tg = groups_from_cutoffs(cut_off_points)
    san = sanity_evaluate(sim_vt, vg, tg)
    if sim_vt_for_v2t is not None:
        san_v = sanity_evaluate(sim_vt_for_v2t, vg, tg)
        for k in ("v2t", "v2t_optimistic", "v2t_ranks"):
            san[k] = san_v[k]
    return {
        "official": off,
        "sanity": {k: san[k] for k in ("t2v", "v2t", "t2v_optimistic", "v2t_optimistic")},
        "t2v_ranks": san["t2v_ranks"].tolist(), "v2t_ranks": san["v2t_ranks"].tolist(),
        "primary": 0.5 * (off["t2v"]["R1"] + off["v2t"]["R1"]),
    }
