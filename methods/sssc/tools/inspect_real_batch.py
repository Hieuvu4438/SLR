from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from method1.config import load_config
from method1.data import Method1Collator, Method1Dataset
from method1.token_spans import tokenize_with_spans
from method1.upstream import create_upret_tokenizer
from method1.utils import atomic_json_dump, sha256_file


def _item_report(dataset: Method1Dataset, item: dict[str, Any], encoded: Any) -> dict[str, Any]:
    video = dataset.videos[item["video_uid"]]
    return {
        "video_uid": item["video_uid"],
        "text_uid": item["text_uid"],
        "group_uid": item["group_uid"],
        "group_member_count": len(
            next(group for group in dataset.groups if group.group_uid == item["group_uid"]).video_uids
        ),
        "full_bpe_count": len(encoded.full_bpe_ids),
        "fully_retained": encoded.fully_retained,
        "input_ids": list(encoded.input_ids),
        "text_valid_count": int(sum(encoded.text_valid)),
        "lexical_span_count": len(encoded.lexical_spans),
        "video_feature_shape": list(item["video_features"].shape),
        "valid_selected_feature_count": int((item["selected_feature_indices"] >= 0).sum()),
        "selected_feature_indices": item["selected_feature_indices"].tolist(),
        "agnostic_hash_reverified": sha256_file(video.agnostic_path) == video.agnostic_sha256,
        "aware_hash_reverified": sha256_file(video.aware_path) == video.aware_sha256,
    }


def inspect(config_path: str, output: str, upret_root: str) -> dict[str, Any]:
    config = load_config(config_path)
    tokenizer = create_upret_tokenizer(upret_root, config.model.bpe_path)
    dataset = Method1Dataset(
        config, split="train", tokenizer=tokenizer, augment=False, auxiliary_cache=None
    )
    candidates = []
    for index, group in enumerate(dataset.groups):
        text = dataset.texts[group.text_uid]
        encoded = tokenize_with_spans(
            text.raw_text,
            text_uid=text.text_uid,
            tokenizer=tokenizer,
            max_positions=config.data.text_max_positions,
        )
        candidates.append((index, encoded))
    short_index, short_encoded = min(candidates, key=lambda value: len(value[1].full_bpe_ids))
    overlong = [
        value
        for value in candidates
        if len(value[1].full_bpe_ids) > config.data.text_max_positions - 2
    ]
    if not overlong:
        raise RuntimeError("real-batch inspection found no overlong training caption")
    long_index, long_encoded = max(overlong, key=lambda value: len(value[1].full_bpe_ids))
    short_item = dataset[short_index]
    long_item = dataset[long_index]
    batch = Method1Collator()((short_item, long_item))
    if long_encoded.fully_retained:
        raise AssertionError("overlong caption was not disabled for auxiliary eligibility")
    report = {
        "schema_version": 1,
        "status": "pass",
        "dataset": config.data.dataset,
        "split": "train",
        "config_sha256": config.digest,
        "manifest_meta_sha256": sha256_file(
            Path(config.data.manifest_dir) / "manifest_meta.json"
        ),
        "short": _item_report(dataset, short_item, short_encoded),
        "overlong": _item_report(dataset, long_item, long_encoded),
        "batch": {
            "video_features_shape": list(batch["video_features"].shape),
            "input_ids_shape": list(batch["input_ids"].shape),
            "class_mask_ignored": bool(batch["video_ignore_raw"][:, 0].all()),
            "finite_features": bool(batch["video_features"].isfinite().all()),
        },
        "limits": {
            "text_positions": config.data.text_max_positions,
            "feature_positions": config.data.feature_len,
        },
    }
    if not all(
        (
            report["short"]["agnostic_hash_reverified"],
            report["short"]["aware_hash_reverified"],
            report["overlong"]["agnostic_hash_reverified"],
            report["overlong"]["aware_hash_reverified"],
            report["batch"]["class_mask_ignored"],
            report["batch"]["finite_features"],
        )
    ):
        raise AssertionError("real-batch feature/hash/mask validation failed")
    atomic_json_dump(report, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upret-root", default="third_party/UPRet")
    args = parser.parse_args()
    print(
        json.dumps(
            inspect(args.config, args.output, args.upret_root),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
