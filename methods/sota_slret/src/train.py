"""Single-GPU trainer for SEDS-based experiments (baseline + methods).

- upstream SEDS model/dataloaders/similarity code imported untouched (src/upstream.py)
- FastCLIP4Clip: numerically-equivalent vectorised clip gather (tests/test_fast_seds.py)
- gradcache: exact emulation of upstream 8x16 DDP contrastive batch (tests/test_gradcache.py)
- selection ONLY on the held-out validation split carved from train (never official test)

Usage:
  python train.py --dataset ph --split_dir <split> --out <run_dir> --seed 42 --epochs 200 \
      [--method baseline] [--method_cfg '{"k": v}'] [-- <extra upstream args>]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upstream import load_upstream, seds_argv  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ph", "csl", "h2s"])
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--method_cfg", default="{}")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--init_model", default=None)
    ap.add_argument("--deterministic", action="store_true", help="upstream cudnn.deterministic")
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return a, extra


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_optimizer(up, args, model, total_steps):
    """Same parameter grouping / BertAdam settings as upstream prep_optimizer (minus DDP wrap)."""
    named = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    is_clip = lambda n: ("clip." in n) or ("clip_rgb." in n)
    is_sign = lambda n: "signbert." in n
    dec = [(n, p) for n, p in named if not any(nd in n for nd in no_decay)]
    nodec = [(n, p) for n, p in named if any(nd in n for nd in no_decay)]
    wd = 0.001
    lr_clip, lr_sign = args.lr * args.coef_lr, args.sign_lr * args.coef_lr
    groups = [
        {'params': [p for n, p in dec if is_clip(n)], 'weight_decay': wd, 'lr': lr_clip},
        {'params': [p for n, p in dec if is_sign(n)], 'weight_decay': wd, 'lr': lr_sign},
        {'params': [p for n, p in dec if not is_clip(n) and not is_sign(n)], 'weight_decay': wd},
        {'params': [p for n, p in nodec if is_clip(n)], 'weight_decay': 0.0, 'lr': lr_clip},
        {'params': [p for n, p in nodec if is_sign(n)], 'weight_decay': 0.0, 'lr': lr_sign},
        {'params': [p for n, p in nodec if not is_clip(n) and not is_sign(n)], 'weight_decay': 0.0},
    ]
    assert sum(len(g['params']) for g in groups) == len(named)
    return up.BertAdam(groups, lr=args.sign_lr, warmup=args.warmup_proportion, schedule='warmup_cosine',
                       b1=0.9, b2=0.98, e=1e-6, t_total=total_steps, weight_decay=wd, max_grad_norm=1.0)


def main():
    a, extra = parse()
    os.makedirs(a.out, exist_ok=True)
    argv = seds_argv(a.dataset, a.split_dir, ["--do_train", "--seed", str(a.seed), "--epochs", str(a.epochs),
                                              "--batch_size", str(a.batch), "--num_thread_reader", str(a.workers),
                                              "--output_dir", a.out] + extra)
    if a.init_model:
        argv += ["--init_model", a.init_model]
    up, args = load_upstream(argv)
    import numpy as np
    import torch
    from modules.tokenization_clip import SimpleTokenizer
    import methods_registry
    from evaluation import encode_eval_set, score_matrices, full_metrics
    from gradcache import gradcache_step

    args = up.set_seed_logger(args)
    device, _ = up.init_device(args, 0)
    if not a.deterministic:
        # NB: cudnn.benchmark is NOT used: pose lengths vary per batch => re-autotune every step.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    method = methods_registry.get(a.method, json.loads(a.method_cfg))
    cache = os.path.join(str(up.PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    state = None
    if a.init_model:
        state = torch.load(a.init_model, map_location="cpu")
        args.init_sign_model = None
    model = method.model_cls.from_pretrained(args.cross_model, cache_dir=cache, distributed=False,
                                             state_dict=state, task_config=args).to(device)
    method.setup(model, args)
    tok = SimpleTokenizer()
    train_dl, n_train, _ = up.DATALOADER_DICT[args.datatype]["train"](args, tok)
    train_dl = method.wrap_train_loader(train_dl, args)
    val_dl, n_val = up.DATALOADER_DICT[args.datatype]["test"](args, tok)  # = held-out split (see split.json)
    val_dl = method.wrap_eval_loader(val_dl, args)
    total_steps = len(train_dl) * a.epochs
    opt = build_optimizer(up, args, model, total_steps)

    try:
        commit = subprocess.check_output(["git", "-C", "/home/haipd/SLR", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "-C", "/home/haipd/SLR", "status", "--porcelain", "methods/sota_slret"]).decode()
    except Exception:
        commit, dirty = "unknown", ""
    split_manifest = json.load(open(os.path.join(a.split_dir, "split.json")))
    assert "official test is never read" in split_manifest["note"]
    cfg = {"wrapper": vars(a), "upstream_args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                 for k, v in vars(args).items()},
           "git_commit": commit, "git_dirty_method_files": dirty, "n_train": n_train, "n_val": n_val,
           "split_val_sha256": split_manifest["val_pkl_sha256"], "method_cfg": method.cfg,
           "selection_rule": "max over epochs of val PrimaryDev = mean(T2V R@1, V2T R@1) (official metric)"}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)

    log = open(os.path.join(a.out, "metrics.jsonl"), "a")
    best, step, start_epoch = -1.0, 0, 0
    resume_path = os.path.join(a.out, "resume.pt")
    if os.path.exists(resume_path):  # crash/session-loss recovery: continue from last finished epoch
        ck = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        best, step, start_epoch = ck["best"], ck["step"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"]); torch.cuda.set_rng_state(ck["rng_cuda"])
        np.random.set_state(ck["rng_np"]); import random as _r; _r.setstate(ck["rng_py"])
        if hasattr(method, "gen") and ck.get("rng_method") is not None:
            method.gen.set_state(ck["rng_method"])
        print(f"resumed from epoch {ck['epoch']} (best {best:.2f})", flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, a.epochs):
        model.train()
        method.on_epoch_start(model, epoch)
        t0, agg = time.time(), {}
        for bi, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            logs = gradcache_step(model, batch, a.chunk, loss_fn=method.loss_fn, extra_loss=method.extra_loss,
                                  n_ranks=-(-a.batch // a.chunk))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            method.after_step(model, batch, step)
            step += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            if not np.isfinite(logs["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {bi}: {logs}")
            if a.max_steps and step >= a.max_steps:
                break
        rec = {"epoch": epoch, "step": step, "train_time_s": time.time() - t0,
               "train": {k: v / (bi + 1) for k, v in agg.items()},
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "logit_scale": float(model.clip.logit_scale.exp())}
        if (epoch + 1) % a.eval_every == 0 or epoch == a.epochs - 1:
            t1 = time.time()
            vids, texts, cut = encode_eval_set(model, val_dl, device)
            sims = score_matrices(model, vids, texts, dual_mix=args.dual_mix, streams=method.eval_streams)
            sim = method.final_scores(sims, model=model, vids=vids, texts=texts)
            sim_v2t = None
            if isinstance(sim, tuple):  # (scores for T2V, scores for V2T)
                sim, sim_v2t = sim
            m = full_metrics(sim, cut, sim_v2t)
            rec.update({"val": {"t2v": m["official"]["t2v"], "v2t": m["official"]["v2t"], "primary": m["primary"],
                                "sanity": m["sanity"]}, "eval_time_s": time.time() - t1})
            if m["primary"] > best:
                best = m["primary"]
                torch.save(model.state_dict(), os.path.join(a.out, "best_model.bin"))
                np.save(os.path.join(a.out, "best_val_sim.npy"), sim.astype(np.float32))
                if sim_v2t is not None:
                    np.save(os.path.join(a.out, "best_val_sim_v2t.npy"), sim_v2t.astype(np.float32))
                if getattr(method, "last_eval_artifacts", None):
                    np.savez(os.path.join(a.out, "best_val_rerank.npz"), **method.last_eval_artifacts)
                for k, v in sims.items():
                    np.save(os.path.join(a.out, f"best_val_sim_stream_{k}.npy"), np.asarray(v, dtype=np.float32))
                json.dump({"epoch": epoch, "primary": best, "t2v_ranks": m["t2v_ranks"], "v2t_ranks": m["v2t_ranks"],
                           "official": m["official"], "sanity": m["sanity"]}, open(os.path.join(a.out, "best_val.json"), "w"))
            rec["best_primary"] = best
        log.write(json.dumps(rec) + "\n"); log.flush()
        v = rec.get("val", {})
        print(f"[ep {epoch}] loss {rec['train'].get('loss', 0):.4f} "
              f"T2V {v.get('t2v', {}).get('R1', -1):.2f} V2T {v.get('v2t', {}).get('R1', -1):.2f} best {best:.2f} "
              f"({rec['train_time_s']:.0f}s, {rec['peak_mem_gb']:.1f}GB)", flush=True)
        import random as _r
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best,
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(), "rng_np": np.random.get_state(),
                    "rng_py": _r.getstate(), "rng_method": method.gen.get_state() if hasattr(method, "gen") else None},
                   resume_path + ".tmp")
        os.replace(resume_path + ".tmp", resume_path)
        if a.save_every and (epoch + 1) % a.save_every == 0:
            torch.save(model.state_dict(), os.path.join(a.out, f"model_ep{epoch}.bin"))
        if a.max_steps and step >= a.max_steps:
            break
    torch.save(model.state_dict(), os.path.join(a.out, "last_model.bin"))
    json.dump({"best_primary": best, "done": True}, open(os.path.join(a.out, "DONE.json"), "w"))
    if os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == "__main__":
    main()
