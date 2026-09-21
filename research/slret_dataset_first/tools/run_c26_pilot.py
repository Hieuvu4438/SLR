"""C26-A frozen Uni-Sign CSL feature cache and fixed contrastive pilot.

The pause supervisor owns terminal job status and restores UniFormerV2. This
worker never reads TEST, never fine-tunes the donor, and writes resumable
per-shard frozen features before fitting only the two linear projections.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from methods.translation_retrieval.bridge import FrozenUniSignDualEncoder
from methods.translation_retrieval.cached import ProjectionRetrievalModel
from methods.translation_retrieval.csl_data import UniSignCSLPoseDataset, collate_pose_batch
from shared.slr_common.evaluation.cico_eval import evaluate_score_matrix


DATA = Path("/home/dongvk/datasets/CSL_Daily_Sentence_Crop")
ASSETS = ROOT / "artifacts/pretrained/c26_unisign_csl"
SHARD_SIZE = 128
INFERENCE_BATCH = 8
MAX_LENGTH = 256
VRAM_LIMIT_BYTES = 20_000_000_000
FREE_DISK_FLOOR = 20 * 1024**3
SEED = 42


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path: Path, value: dict) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class Progress:
    def __init__(self, job_dir: Path, total: int) -> None:
        self.path = job_dir / "status.json"
        self.state = json.loads(self.path.read_text())
        self.state.update(worker_pid=os.getpid(), status="RUNNING", total=total,
                          completed=0, stage="load_donor")
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        self.update()

    def update(self, **changes: object) -> None:
        with self.lock:
            self.state.update(changes)
            self.state["updated_at"] = stamp()
            if self.state["completed"] and self.state["stage"].startswith("extract"):
                rate = self.state["completed"] / max(time.monotonic() - self.started, 1)
                self.state["eta_seconds"] = (self.state["total"] - self.state["completed"]) / rate
            atomic_json(self.path, self.state)

    def heartbeat(self) -> None:
        while not self.stop.wait(30):
            self.update()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def group_id(name: str) -> str:
    return name.split("_", 1)[0]


def load_donor() -> FrozenUniSignDualEncoder:
    donor_root = ROOT / "third_party/Uni-Sign"
    sys.path.insert(0, str(donor_root))
    import config  # type: ignore[import-not-found]
    config.mt5_path = str(ASSETS / "mt5-base")
    from models import Uni_Sign  # type: ignore[import-not-found]
    from types import SimpleNamespace

    donor = Uni_Sign(SimpleNamespace(hidden_dim=256, dataset="CSL_Daily", rgb_support=False))
    checkpoint = torch.load(ASSETS / "unisign/csl_daily_pose_only_slt.pth",
                            map_location="cpu", weights_only=True, mmap=True)
    result = donor.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Uni-Sign checkpoint did not strict-load")
    bridge = FrozenUniSignDualEncoder(donor).eval().cuda()
    return bridge


def check_resources(job_dir: Path) -> int:
    if shutil.disk_usage(job_dir).free < FREE_DISK_FLOOR + 1024**3:
        raise RuntimeError("C26 free-space floor would be crossed")
    peak = torch.cuda.max_memory_reserved()
    if peak > VRAM_LIMIT_BYTES:
        raise RuntimeError(f"C26 process VRAM limit exceeded: {peak} bytes")
    return peak


def extract_split(split: str, bridge: FrozenUniSignDualEncoder, job_dir: Path,
                  progress: Progress, completed_before: int) -> Path:
    dataset = UniSignCSLPoseDataset(DATA / f"labels.{split}", DATA / "keypoint",
                                    split=split, max_length=MAX_LENGTH, seed=SEED)
    if split == "train":
        dataset.set_epoch(0)
    target = job_dir / "features" / split
    target.mkdir(parents=True, exist_ok=True)
    for begin in range(0, len(dataset), SHARD_SIZE):
        end = min(begin + SHARD_SIZE, len(dataset))
        path = target / f"{begin:05d}-{end:05d}.pt"
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["split"] != split or cached["begin"] != begin or cached["end"] != end:
                raise RuntimeError(f"cache identity mismatch: {path}")
            if len(cached["names"]) != end - begin or cached["video"].shape != (end-begin, 768):
                raise RuntimeError(f"cache shape mismatch: {path}")
            progress.update(stage=f"extract_{split}", completed=completed_before + end,
                            peak_cuda_bytes=check_resources(job_dir))
            continue
        loader = DataLoader(Subset(dataset, range(begin, end)), batch_size=INFERENCE_BATCH,
                            shuffle=False, num_workers=0, collate_fn=collate_pose_batch)
        names: list[str] = []
        captions: list[str] = []
        video_parts: list[torch.Tensor] = []
        text_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch_names, parts, mask, batch_captions in loader:
                cuda_parts = {key: value.cuda(non_blocking=True) for key, value in parts.items()}
                video = bridge.video_features(cuda_parts, mask.cuda(non_blocking=True))
                text = bridge.text_features(batch_captions)
                if video.shape != (len(batch_names), 768) or text.shape != video.shape:
                    raise RuntimeError("C26 frozen feature width/batch mismatch")
                if not torch.isfinite(video).all() or not torch.isfinite(text).all():
                    raise RuntimeError("C26 frozen feature contains nonfinite values")
                video_parts.append(video.cpu())
                text_parts.append(text.cpu())
                names.extend(batch_names)
                captions.extend(batch_captions)
                check_resources(job_dir)
        atomic_torch(path, {"split": split, "begin": begin, "end": end,
                            "names": names, "captions": captions,
                            "video": torch.cat(video_parts), "text": torch.cat(text_parts),
                            "max_length": MAX_LENGTH})
        progress.update(stage=f"extract_{split}", completed=completed_before + end,
                        peak_cuda_bytes=check_resources(job_dir))
        print(f"{stamp()} {split} {end}/{len(dataset)} peak_reserved={torch.cuda.max_memory_reserved()}",
              flush=True)
    return target


def load_features(directory: Path, expected: int) -> dict:
    chunks = [torch.load(path, map_location="cpu", weights_only=True)
              for path in sorted(directory.glob("*.pt"))]
    if not chunks or chunks[0]["begin"] != 0 or chunks[-1]["end"] != expected:
        raise RuntimeError(f"incomplete C26 cache: {directory}")
    if any(a["end"] != b["begin"] for a, b in zip(chunks, chunks[1:])):
        raise RuntimeError(f"gapped C26 cache: {directory}")
    result = {key: sum((chunk[key] for chunk in chunks), []) for key in ("names", "captions")}
    result["video"] = torch.cat([chunk["video"] for chunk in chunks])
    result["text"] = torch.cat([chunk["text"] for chunk in chunks])
    if len(result["names"]) != expected or len(set(result["names"])) != expected:
        raise RuntimeError("C26 cache IDs are missing or duplicate")
    if not torch.isfinite(result["video"]).all() or not torch.isfinite(result["text"]).all():
        raise RuntimeError("C26 cache contains nonfinite values")
    return result


def dev_gallery(features: dict) -> tuple[torch.Tensor, list[str], dict[str, list[str]], dict[str, list[str]]]:
    names: list[str] = features["names"]
    captions: list[str] = features["captions"]
    groups: dict[str, list[str]] = defaultdict(list)
    first_index: dict[str, int] = {}
    group_text: dict[str, str] = {}
    for index, (name, caption) in enumerate(zip(names, captions, strict=True)):
        gid = group_id(name)
        groups[gid].append(name)
        first_index.setdefault(gid, index)
        if gid in group_text and group_text[gid] != caption:
            raise RuntimeError(f"caption changed within CSL group {gid}")
        group_text[gid] = caption
    text_ids = sorted(groups)
    text = features["text"][[first_index[gid] for gid in text_ids]]
    video_to_text = {name: [group_id(name)] for name in names}
    text_to_video = {gid: groups[gid] for gid in text_ids}
    if len(names) != 1077 or len(text_ids) != 797:
        raise RuntimeError("unexpected CSL DEV gallery size")
    return text, text_ids, video_to_text, text_to_video


def evaluate(scores: torch.Tensor, features: dict, text_ids: list[str],
             v2t: dict[str, list[str]], t2v: dict[str, list[str]],
             job_dir: Path, label: str) -> dict:
    matrix = scores.detach().float().cpu().numpy()
    np.save(job_dir / f"scores_{label}.npy", matrix)
    result = evaluate_score_matrix(matrix, video_ids=features["names"],
                                   text_ids=text_ids, video_to_text=v2t,
                                   text_to_video=t2v)
    result.pop("per_query")
    result["mean_r1"] = (result["V2T"]["R1"] + result["T2V"]["R1"]) / 2
    atomic_json(job_dir / f"metrics_{label}.json", result)
    print(f"{stamp()} {label} T2V_R1={result['T2V']['R1']:.4f} "
          f"V2T_R1={result['V2T']['R1']:.4f} mean_R1={result['mean_r1']:.4f}", flush=True)
    return result


def unique_group_batches(names: list[str], epoch: int, batch_size: int = 256):
    rng = random.Random(SEED + epoch)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        by_group[group_id(name)].append(index)
    group_ids = list(by_group)
    rng.shuffle(group_ids)
    for indices in by_group.values():
        rng.shuffle(indices)
    for round_index in range(max(map(len, by_group.values()))):
        round_items = [by_group[gid][round_index] for gid in group_ids
                       if round_index < len(by_group[gid])]
        for begin in range(0, len(round_items), batch_size):
            batch = round_items[begin:begin + batch_size]
            if len(batch) >= 2:
                yield batch


def train_head(train: dict, dev: dict, job_dir: Path, progress: Progress) -> dict:
    text, text_ids, v2t, t2v = dev_gallery(dev)
    zero_scores = F.normalize(dev["video"], dim=1) @ F.normalize(text, dim=1).T
    zero = evaluate(zero_scores, dev, text_ids, v2t, t2v, job_dir, "zero_shot")

    torch.manual_seed(SEED)
    model = ProjectionRetrievalModel(768, 768, output_dim=256).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    records = {"zero_shot": zero["mean_r1"], "epochs": []}
    best_score = -1.0
    for epoch in range(1, 6):
        model.train()
        losses: list[float] = []
        updates = 0
        for batch in unique_group_batches(train["names"], epoch):
            indexes = torch.tensor(batch, dtype=torch.long)
            video = train["video"][indexes].cuda(non_blocking=True)
            caption = train["text"][indexes].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(video, caption)
            if not torch.isfinite(loss):
                raise RuntimeError("C26 projection loss nonfinite")
            loss.backward()
            if any(param.grad is None or not torch.isfinite(param.grad).all()
                   for param in model.parameters()):
                raise RuntimeError("C26 projection gradient missing/nonfinite")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            updates += 1
        model.eval()
        with torch.inference_mode():
            video_dev = dev["video"].cuda()
            text_dev = text.cuda()
            scores = model.score_gallery(video_dev, text_dev)
            result = evaluate(scores, dev, text_ids, v2t, t2v, job_dir, f"epoch{epoch}")
        record = {"epoch": epoch, "updates": updates,
                  "train_loss_mean": sum(losses) / len(losses),
                  "mean_r1": result["mean_r1"],
                  "T2V_R1": result["T2V"]["R1"], "V2T_R1": result["V2T"]["R1"]}
        records["epochs"].append(record)
        progress.update(stage="train_head", epoch=epoch, loss=record["train_loss_mean"],
                        dev_mean_r1=record["mean_r1"], peak_cuda_bytes=check_resources(job_dir))
        if result["mean_r1"] > best_score:
            best_score = result["mean_r1"]
            atomic_torch(job_dir / "best_head.pt", {"model": model.state_dict(),
                         "epoch": epoch, "metrics": record, "seed": SEED})
    records["selected_mean_r1"] = best_score
    records["selected_epoch"] = max(records["epochs"], key=lambda row: row["mean_r1"])["epoch"]
    records["gallery"] = {"videos": len(dev["names"]), "texts": len(text_ids)}
    atomic_json(job_dir / "training.json", records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--feature-source", type=Path,
                        help="completed frozen-feature cache from a prior technical attempt")
    args = parser.parse_args()
    job_dir = args.job_dir.resolve()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_count, dev_count = 18401, 1077
    progress = Progress(job_dir, train_count + dev_count)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("C26 CUDA device unavailable")
        if shutil.disk_usage(job_dir).free < FREE_DISK_FLOOR + 1024**3:
            raise RuntimeError("C26 disk admission failed")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        if args.feature_source is None:
            bridge = load_donor()
            progress.update(stage="extract_dev", peak_cuda_bytes=check_resources(job_dir))
            dev_dir = extract_split("dev", bridge, job_dir, progress, 0)
            train_dir = extract_split("train", bridge, job_dir, progress, dev_count)
            del bridge
            torch.cuda.empty_cache()
        else:
            source = args.feature_source.resolve()
            dev_dir, train_dir = source / "dev", source / "train"
            if not dev_dir.is_dir() or not train_dir.is_dir():
                raise RuntimeError("C26 prior feature cache is missing TRAIN/DEV")
            progress.update(stage="reuse_frozen_features", completed=train_count+dev_count,
                            feature_source=str(source))
        progress.update(stage="load_cached_features", completed=train_count+dev_count)
        dev = load_features(dev_dir, dev_count)
        train = load_features(train_dir, train_count)
        result = train_head(train, dev, job_dir, progress)
        atomic_json(job_dir / "worker_summary.json", {
            "status": "COMPLETED", "stage": "pilot_measured", "run_id": job_dir.name,
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "records": result, "test_loaded": False,
            "donor_frozen": True, "train_count": train_count, "dev_count": dev_count,
        })
        progress.update(stage="resume_uniformerv2", completed=train_count+dev_count,
                        dev_mean_r1=result["selected_mean_r1"])
    finally:
        progress.close()


if __name__ == "__main__":
    main()
