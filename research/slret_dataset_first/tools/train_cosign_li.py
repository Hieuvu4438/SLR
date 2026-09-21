#!/usr/bin/env python3
"""Run one frozen-feature CoSign-LI or matched mean-pooling arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/SEDS"))

from methods.sl_mvr import DualLevelLateInteraction
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim


ANNOTATION_ROOT = Path(
    "/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/"
    "PHOENIX-2014-T/annotations/manual"
)
DEFAULT_SIGNREP = ROOT / "artifacts/slret_goal_v2/signrep-native-window-pilot-001"
DEFAULT_TEXT = ROOT / "artifacts/slret_goal/dataset-first-text-e5-001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_ids(split: str) -> list[str]:
    path = ANNOTATION_ROOT / f"PHOENIX-2014-T.{split}.corpus.csv"
    with path.open(encoding="utf-8") as stream:
        return [row["name"] for row in csv.DictReader(stream, delimiter="|")]


class FrozenBank:
    def __init__(self, signrep_root: Path, text_root: Path, split: str) -> None:
        available = {path.stem for path in (signrep_root / split).glob("*.npz")}
        self.ids = [sample_id for sample_id in annotation_ids(split) if sample_id in available]
        if not self.ids:
            raise ValueError(f"no cached examples for {split}")
        self.rows = []
        for sample_id in self.ids:
            visual_path = signrep_root / split / f"{sample_id}.npz"
            text_path = text_root / split / f"{sample_id}.npz"
            if not text_path.is_file():
                raise FileNotFoundError(text_path)
            with np.load(visual_path) as data:
                features = torch.from_numpy(data["features"].astype(np.float32))
                latents = torch.from_numpy(data["latent"].astype(np.float32))
            with np.load(text_path) as data:
                text = torch.from_numpy(data["tokens"].astype(np.float32))
            if features.ndim != 2 or latents.shape != features.shape or text.ndim != 2:
                raise ValueError(f"invalid cache shape for {sample_id}")
            if not torch.isfinite(features).all() or not torch.isfinite(latents).all() or not torch.isfinite(text).all():
                raise ValueError(f"non-finite cache for {sample_id}")
            self.rows.append((features, latents, text))

    def batch(self, indices: list[int], device: str):
        selected = [self.rows[index] for index in indices]
        features = pad_sequence([row[0] for row in selected], batch_first=True)
        latents = pad_sequence([row[1] for row in selected], batch_first=True)
        texts = pad_sequence([row[2] for row in selected], batch_first=True)
        video_mask = torch.arange(features.shape[1])[None, :] < torch.tensor(
            [row[0].shape[0] for row in selected]
        )[:, None]
        text_mask = torch.arange(texts.shape[1])[None, :] < torch.tensor(
            [row[2].shape[0] for row in selected]
        )[:, None]
        return tuple(value.to(device) for value in (features, latents, texts, video_mask, text_mask))


def fixed_batches(size: int, batch_size: int, updates: int, seed: int) -> list[list[int]]:
    generator = random.Random(seed)
    batches: list[list[int]] = []
    order: list[int] = []
    while len(batches) < updates:
        order = list(range(size))
        generator.shuffle(order)
        for offset in range(0, size - batch_size + 1, batch_size):
            batches.append(order[offset : offset + batch_size])
            if len(batches) == updates:
                break
    return batches


@torch.inference_mode()
def evaluate(model, bank: FrozenBank, interaction: str, device: str, chunk: int):
    model.eval()
    count = len(bank.ids)
    scores = np.empty((count, count), dtype=np.float32)
    video_chunks = []
    text_chunks = []
    for start in range(0, count, chunk):
        indices = list(range(start, min(start + chunk, count)))
        features, latents, texts, video_mask, text_mask = bank.batch(indices, device)
        feature_video, latent_video = model.encode_video(features, latents)
        feature_text, latent_text = model.encode_text(texts)
        video_chunks.append((start, feature_video, latent_video, video_mask))
        text_chunks.append((start, feature_text, latent_text, text_mask))
    for video_start, feature_video, latent_video, video_mask in video_chunks:
        for text_start, feature_text, latent_text, text_mask in text_chunks:
            output = model.score_encoded(
                feature_video, latent_video, feature_text, latent_text,
                video_mask, text_mask, interaction=interaction,
            )["scores"]
            scores[
                video_start : video_start + feature_video.shape[0],
                text_start : text_start + feature_text.shape[0],
            ] = output.cpu().numpy()
    v2t = tensor_text_to_video_metrics(scores[:, None, :])
    t2v = compute_metrics(tensor_video_to_text_sim(scores[:, None, :]))
    metrics = {
        "T2V": {key: value for key, value in t2v.items() if key != "cols"},
        "V2T": {key: value for key, value in v2t.items() if key != "cols"},
    }
    metrics["mean_R1"] = 0.5 * (metrics["T2V"]["R1"] + metrics["V2T"]["R1"])
    return scores, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction", choices=("mean", "late"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signrep-root", type=Path, default=DEFAULT_SIGNREP)
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--eval-at", type=int, nargs="+", default=(0, 80, 160))
    parser.add_argument("--eval-chunk", type=int, default=16)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    if sorted(set(args.eval_at)) != sorted(args.eval_at) or 0 not in args.eval_at or args.updates not in args.eval_at:
        raise ValueError("eval-at must be unique and contain 0 and updates")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train = FrozenBank(args.signrep_root, args.text_root, "train")
    dev = FrozenBank(args.signrep_root, args.text_root, "dev")
    visual_dim = train.rows[0][0].shape[-1]
    text_dim = train.rows[0][2].shape[-1]
    model = DualLevelLateInteraction(
        visual_dim=visual_dim,
        text_dim=text_dim,
        projection_dim=args.projection_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    batches = fixed_batches(len(train.ids), args.batch_size, args.updates, args.seed)
    (args.output / "batch_indices.json").write_text(json.dumps(batches) + "\n")

    started = time.time()
    history = []
    best = None
    for step in range(args.updates + 1):
        if step in args.eval_at:
            matrix, metrics = evaluate(model, dev, args.interaction, args.device, args.eval_chunk)
            evaluation = args.output / f"eval_step{step:04d}"
            evaluation.mkdir()
            np.save(evaluation / "scores_video_x_text.npy", matrix)
            (evaluation / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            record = {"step": step, "metrics": metrics}
            history.append(record)
            if best is None or metrics["mean_R1"] > best["metrics"]["mean_R1"]:
                best = record
                torch.save(model.state_dict(), args.output / "best.pt")
            print(json.dumps({"event": "evaluation", **record}), flush=True)
        if step == args.updates:
            break
        model.train()
        batch = train.batch(batches[step], args.device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(*batch, interaction=args.interaction)
        loss = model.loss(outputs)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(json.dumps({"event": "train", "step": step + 1, "loss": float(loss)}), flush=True)

    report = {
        "run_id": args.output.name,
        "status": "complete",
        "interaction": args.interaction,
        "seed": args.seed,
        "train_rows": len(train.ids),
        "dev_rows": len(dev.ids),
        "train_ids_sha256": hashlib.sha256("\n".join(train.ids).encode()).hexdigest(),
        "dev_ids_sha256": hashlib.sha256("\n".join(dev.ids).encode()).hexdigest(),
        "signrep_run_sha256": sha256(args.signrep_root.parent / "c16-signrep-assets-001/run.json"),
        "text_run_sha256": sha256(args.text_root / "run.json"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "projection_dim": args.projection_dim,
        "history": history,
        "selected": best,
        "wall_seconds": time.time() - started,
        "test_used": False,
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
