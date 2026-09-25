"""Re-score the VAL split with a trained checkpoint under probe conditions (never official test).

python probe_eval.py --dataset ph --split_dir <split> --ckpt <run>/best_model.bin --out <npz>
  [--method baseline --method_cfg '{}'] [--shuffle_clips]  (Probe C: permute valid clip order, pose+RGB jointly)
Saves per-stream val score matrices [N_video, N_text_group] + cut_off_points.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/haipd/SLR/methods/sota_slret/src")
from upstream import load_upstream, seds_argv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--method_cfg", default="{}")
    ap.add_argument("--shuffle_clips", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    up, args = load_upstream(seds_argv(a.dataset, a.split_dir, ["--do_eval", "--num_thread_reader", "8",
                                                               "--output_dir", os.path.dirname(os.path.abspath(a.out))]))
    import numpy as np
    import torch
    import methods_registry
    from evaluation import score_matrices, full_metrics
    from modules.tokenization_clip import SimpleTokenizer
    args = up.set_seed_logger(args)
    device, _ = up.init_device(args, 0)
    method = methods_registry.get(a.method, json.loads(a.method_cfg))
    cache = os.path.join(str(up.PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    args.init_sign_model = None
    model = method.model_cls.from_pretrained(args.cross_model, cache_dir=cache, distributed=False, state_dict=None,
                                             task_config=args).to(device)
    method.setup(model, args)
    missing, unexpected = model.load_state_dict(torch.load(a.ckpt, map_location="cpu"), strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.eval()
    dl, _ = up.DATALOADER_DICT[args.datatype]["test"](args, SimpleTokenizer())
    dl = method.wrap_eval_loader(dl, args)
    g = torch.Generator().manual_seed(a.seed)
    cut = [c - 1 for c in dl.dataset.cut_off_points]
    vids, texts, total = [], [], 0
    with torch.no_grad():
        for batch in dl:
            s = {k: v.to(device) for k, v in batch.items()}
            rgb, cs, mask = s['RGB_feature'], s['body_clips_start'], s['body_mask']
            if a.shuffle_clips:
                rgb, cs = rgb.clone(), cs.clone()
                for i in range(rgb.shape[0]):
                    nval = int((cs[i] != -1).sum())
                    perm = torch.randperm(nval, generator=g).to(device)
                    rgb[i, :, :nval] = rgb[i, :, perm]
                    cs[i, :nval] = cs[i, perm]
            body = {'pose': s['body_pose'], 'clips_start': cs, 'mask': mask, 'rgb': rgb}
            m, vp, vr = model.get_visual_output({'pose': s['right_pose']}, {'pose': s['left_pose']}, body, shaped=True, get_hidden=True)
            vids.append((m, vp, vr))
            b = s['pairs_text'].shape[0]
            keep = [i - total for i in cut if total <= i < total + b]
            if keep:
                texts.append(model.get_sequence_output(s['pairs_text'][keep], s['pairs_segment'][keep], s['pairs_mask'][keep], get_hidden=True))
            total += b
    sims = score_matrices(model, vids, texts, dual_mix=args.dual_mix)
    out = {f"sim_{k}": v.astype(np.float32) for k, v in sims.items()}
    np.savez(a.out, cut=np.array(dl.dataset.cut_off_points), **out)
    for k, v in sims.items():
        m = full_metrics(v, dl.dataset.cut_off_points)
        print(k, "T2V %.2f V2T %.2f" % (m["official"]["t2v"]["R1"], m["official"]["v2t"]["R1"]))


if __name__ == "__main__":
    main()
