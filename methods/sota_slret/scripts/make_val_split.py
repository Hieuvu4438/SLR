"""Carve a seeded, sentence-level held-out validation split from the official TRAIN split.

Why: the SEDS release ships features only for train/test (no dev), and upstream selects
checkpoints on test. Our firewall requires dev-only selection, so we hold out train
sentences. Output layout keeps the upstream dataloaders untouched:

  <out>/data/train.pkl   train minus held-out sentences
  <out>/data/test.pkl    held-out sentences (loaded by upstream as subset="test")
  <out>/rgb/train  -> symlink to original RGB train dir
  <out>/rgb/test/  per-file symlinks to the held-out videos' RGB features
  <out>/split.json manifest (seed, ids, sha256 of both pickles)
"""
import argparse
import hashlib
import json
import os
import pickle as pkl
import random

SEDS = "/home/haipd/SLR/third_party/SEDS"
CFG = {
    # dataset: (annotation dir, RGB feature root, number of held-out sentence keys)
    "ph": ("data_ph", "PHOENIX-2014-T/I3D_features", 519),     # = official dev size
    "csl": ("data_csl", "CSL/I3D_features", 800),              # ~ test sentence count
    "h2s": ("data_h2", "How2Sign/I3D_features", 1500),         # ~ official val size
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def video_names(entry):
    # How2Sign loaders resolve feature files by `new_video_name`; PH/CSL by `video_name`.
    name = lambda e: e.get("new_video_name", e["video_name"])
    return [name(e) for e in entry] if isinstance(entry, list) else [name(entry)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(CFG))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ann, rgb_root, n_val = CFG[a.dataset]
    out = a.out or f"/home/haipd/SLR/artifacts/sota_slret_agent/splits/{a.dataset}_val_s{a.seed}"
    train = pkl.load(open(os.path.join(SEDS, ann, "train.pkl"), "rb"))
    keys = sorted(train)
    rng = random.Random(a.seed)
    val_keys = set(rng.sample(keys, n_val))
    new_train = {k: v for k, v in train.items() if k not in val_keys}
    val = {k: train[k] for k in keys if k in val_keys}

    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    os.makedirs(os.path.join(out, "rgb", "test"), exist_ok=True)
    pkl.dump(new_train, open(os.path.join(out, "data", "train.pkl"), "wb"))
    pkl.dump(val, open(os.path.join(out, "data", "test.pkl"), "wb"))
    src_train = os.path.realpath(os.path.join(SEDS, rgb_root, "train"))
    link = os.path.join(out, "rgb", "train")
    if not os.path.islink(link):
        os.symlink(src_train, link)
    n_vid = 0
    for k, v in val.items():
        for name in video_names(v):
            src = os.path.join(src_train, name + ".pkl")
            assert os.path.exists(src), src
            dst = os.path.join(out, "rgb", "test", name + ".pkl")
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            n_vid += 1
    manifest = {
        "dataset": a.dataset, "seed": a.seed, "source_train": os.path.join(SEDS, ann, "train.pkl"),
        "source_train_sha256": sha256(os.path.join(SEDS, ann, "train.pkl")),
        "n_train_keys": len(new_train), "n_val_keys": len(val), "n_val_videos": n_vid,
        "train_pkl_sha256": sha256(os.path.join(out, "data", "train.pkl")),
        "val_pkl_sha256": sha256(os.path.join(out, "data", "test.pkl")),
        "val_keys": sorted(val),
        "note": "val is carved from official TRAIN only; official test is never read here",
    }
    json.dump(manifest, open(os.path.join(out, "split.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "val_keys"}, indent=1))


if __name__ == "__main__":
    main()
