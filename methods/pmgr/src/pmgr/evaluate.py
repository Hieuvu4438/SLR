from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from pmgr.config import load_config, resolved_with_overrides
from pmgr.data.group_dataset import GroupDataset
from pmgr.metrics import evaluate_protocol
from pmgr.model import PMGRRetriever, build_retriever, load_tokenizer
from pmgr.scoring import mixed_pair_scores
from slr_common.data.tokenize import encode_cico_text
from slr_common.utils import atomic_json_dump


def _chunks(length: int, size: int):
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def _encode_gallery(
    model: PMGRRetriever,
    dataset: GroupDataset,
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[int]]:
    model.eval()
    video_hidden: list[torch.Tensor] = []
    video_valid: list[torch.Tensor] = []
    video_ids: list[str] = []
    owner: list[int] = []
    video_batch_size = int(config["validation"]["video_encoder_batch"])
    group_ids: list[str] = []
    texts: list[str] = []
    all_features: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    with torch.no_grad():
        for group_position in range(len(dataset)):
            item = dataset[group_position]
            group_ids.append(item["group_id"])
            texts.append(item["canonical_text"])
            all_features.append(item["features"])
            all_valid.append(item["valid"])
            video_ids.extend(item["video_ids"])
            owner.extend([group_position] * len(item["video_ids"]))
        packed_features = torch.cat(all_features)
        packed_valid = torch.cat(all_valid)
        for block in _chunks(len(packed_features), video_batch_size):
            h = packed_features[block].transpose(1, 2).unsqueeze(-1).contiguous().to(device)
            valid = packed_valid[block].to(device)
            legacy = torch.ones((len(h), h.shape[2] + 1), dtype=torch.long, device=device)
            legacy[:, 1:] = (~valid).long()
            hidden, normalized_valid = model.encode_video(h, legacy)
            video_hidden.append(hidden.cpu())
            video_valid.append(normalized_valid.cpu())

        encoded_text = [
            encode_cico_text(text, tokenizer, int(config["data"]["max_text_tokens"]))
            for text in texts
        ]
        text_hidden: list[torch.Tensor] = []
        text_valid: list[torch.Tensor] = []
        batch_size = int(config["validation"]["text_encoder_batch"])
        for block in _chunks(len(encoded_text), batch_size):
            values = encoded_text[block]
            ids, segments, masks = (
                torch.stack([value[position] for value in values]).to(device)
                for position in range(3)
            )
            hidden, valid = model.encode_text(ids, segments, masks)
            text_hidden.append(hidden.cpu())
            text_valid.append(valid.cpu())
    return (
        torch.cat(video_hidden), torch.cat(video_valid), torch.cat(text_hidden),
        torch.cat(text_valid), video_ids, group_ids, owner,
    )


def evaluate_model(
    model: PMGRRetriever,
    config: dict[str, Any],
    *,
    split: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if split == "test" and config["paths"].get("test_index") is None:
        raise ValueError("test is locked: no test index is configured")
    index_path = config["paths"][f"{split}_index"]
    dataset = GroupDataset(
        index_path,
        feature_len=int(config["data"]["max_features"]),
        alpha=float(config["data"]["feature_mix_alpha"]),
    )
    expected_groups = config["data"][f"expected_{split}_groups"]
    expected_videos = config["data"][f"expected_{split}_videos"]
    if expected_groups is None or expected_videos is None:
        raise ValueError(f"{split} population counts are not configured")
    actual_population = (dataset.index.group_count, dataset.index.video_count)
    expected_population = (int(expected_groups), int(expected_videos))
    if actual_population != expected_population:
        raise ValueError(
            f"{split} population count mismatch: {actual_population} != {expected_population}"
        )
    tokenizer = load_tokenizer(config)
    video, video_valid, text, text_valid, video_ids, group_ids, owner = _encode_gallery(
        model, dataset, tokenizer, config, device
    )
    scores = np.empty((len(video), len(text)), dtype=np.float32)
    scoring = config["scoring"]
    with torch.no_grad():
        for row in _chunks(len(video), int(config["validation"]["score_video_block"])):
            v = video[row].to(device)
            vv = video_valid[row].to(device)
            for column in _chunks(len(text), int(config["validation"]["score_text_block"])):
                t = text[column].to(device)
                tv = text_valid[column].to(device)
                q, _, _ = mixed_pair_scores(
                    v, t, t, vv, tv, tv,
                    omega=float(scoring["dual_mix"]),
                    sigma=float(scoring["inner_temperature"]),
                    normalize_eps=float(scoring["normalize_eps"]),
                    mask_policy=scoring["mask_policy"],
                )
                scores[row, column] = q.cpu().numpy()
    metrics = evaluate_protocol(
        scores, video_ids=video_ids, group_ids=group_ids, video_to_group=owner
    )
    metrics["split"] = split
    return scores, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full-gallery PMGR evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    parser.add_argument("--mask-policy", choices=("valid_tokens_only", "legacy_unmasked"))
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final_test:
        raise ValueError("test evaluation requires the explicit --final-test lock")
    if args.split != "test" and args.final_test:
        raise ValueError("--final-test is valid only for split=test")
    validation_mode = "final_test" if args.split == "test" else "validation"
    config = load_config(args.config, mode=validation_mode)
    config = resolved_with_overrides(
        config,
        validation_mode=validation_mode,
        scoring__mask_policy=args.mask_policy,
    )
    device = torch.device(args.device)
    model, _ = build_retriever(config, checkpoint=args.checkpoint, device=device)
    _, result = evaluate_model(model, config, split=args.split, device=device)
    if args.output:
        atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
