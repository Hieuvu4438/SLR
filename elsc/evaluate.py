from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from elsc.config import load_config
from elsc.data.cico_dataset import CiCoFeatureDataset
from elsc.data.tokenize import CiCoCollator
from elsc.evaluation.cico_eval import evaluate_score_matrix
from elsc.evaluation.runtime import encode_gallery, score_gallery_blockwise
from elsc.provenance import validate_test_lock
from elsc.resources import require_resources
from elsc.upstream.factory import build_retriever_from_checkpoint, load_cico_tokenizer
from elsc.utils import atomic_json_dump, ordered_hash, sha256_file, sha256_json


def evaluate_model(
    model, config: dict, split: str, device: torch.device
) -> tuple[np.ndarray, dict]:
    data = config["data"]
    dataset = CiCoFeatureDataset(
        data[f"{split}_manifest"], feature_len=data["feature_len"], alpha=data["alpha"], split=split
    )
    tokenizer = load_cico_tokenizer(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"].get("eval_batch", config["train"]["per_device_batch"])),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 4)),
        collate_fn=CiCoCollator(tokenizer, int(data["max_words"]), augment=False),
    )
    gallery = encode_gallery(model, loader, device)
    if gallery.video_to_text is None or gallery.text_to_video is None:
        raise RuntimeError("gallery encoder did not preserve positive ID mappings")
    scores = score_gallery_blockwise(
        model.bridge,
        gallery,
        device=device,
        dual_mix=float(config["model"]["dual_mix"]),
        video_block=int(config["evaluation"].get("video_block", 128)),
        text_block=int(config["evaluation"].get("text_block", 128)),
    )
    metrics = evaluate_score_matrix(
        scores,
        video_ids=gallery.video_ids,
        text_ids=gallery.text_ids,
        video_to_text=gallery.video_to_text,
        text_to_video=gallery.text_to_video,
    )
    metrics["split"] = split
    metrics["id_hashes"] = {
        "videos": ordered_hash(gallery.video_ids),
        "texts": ordered_hash(gallery.text_ids),
        "positive_mapping": sha256_json(
            {
                "video_to_text": gallery.video_to_text,
                "text_to_video": gallery.text_to_video,
            }
        ),
    }
    return scores, metrics


def _checkpoint_path(run_dir: Path, value: str) -> Path:
    aliases = {"best_dev": "best_dev.pt", "last": "last.pt"}
    return run_dir / "checkpoints" / aliases.get(value, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full-gallery CiCo/ELSC retrieval evaluation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--checkpoint", default="best_dev")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    config = load_config(run_dir / "resolved_config.yaml", stage="evaluate")
    checkpoint = _checkpoint_path(run_dir, args.checkpoint)
    selection = None
    if args.split == "test":
        selection = validate_test_lock(
            run_dir / "selection.json", checkpoint, run_dir=run_dir, config=config
        )
    resources = config.get("resources", {})
    require_resources(
        run_dir,
        torch.device(args.device),
        min_disk_gib=float(resources.get("min_free_disk_gib", 20)),
        min_gpu_gib=float(resources.get("min_free_gpu_gib_evaluation", 4)),
        operation=f"{args.split} evaluation",
    )
    model, _ = build_retriever_from_checkpoint(config, checkpoint, device=args.device)
    scores, metrics = evaluate_model(model, config, args.split, torch.device(args.device))
    output = run_dir / "evaluation" / args.split
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "scores_video_x_text.npy", scores)
    metrics["checkpoint"] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    if selection is not None:
        metrics["selection_lock"] = {
            "path": str(run_dir / "selection.json"),
            "sha256": sha256_file(run_dir / "selection.json"),
            "selected_epoch": selection["selected_epoch"],
        }
    atomic_json_dump(metrics, output / "metrics.json")
    atomic_json_dump(
        {
            "schema_version": 1,
            "split": args.split,
            "score_orientation": "video_x_text",
            "V2T": metrics["per_query"]["V2T"],
            "T2V": metrics["per_query"]["T2V"],
        },
        output / "per_query_ranks.json",
    )
    print(json.dumps({key: metrics[key] for key in ("V2T", "T2V", "tie_stats")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
