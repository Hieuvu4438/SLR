from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from elsc.config import config_hash, load_config
from elsc.data.cache_dataset import CacheMismatchError, validate_cache_meta
from elsc.data.cico_dataset import CiCoFeatureDataset
from elsc.data.manifest import load_manifest
from elsc.data.tokenize import CiCoCollator, encode_cico_text
from elsc.data.word_offsets import eligible_word_units, encode_with_offsets, replace_canonical_span
from elsc.mining.negative_graph import (
    PrototypeOccurrence,
    build_frequency_matched_random_graph,
    build_visual_neighbor_graph,
    filter_occurrence_negatives,
    raw_support_prototype,
)
from elsc.mining.teacher_align import reliability, select_support, token_word_distributions
from elsc.losses.evidence import receptive_field_closure, select_matched_control
from elsc.upstream.factory import build_retriever_from_checkpoint, load_cico_tokenizer_components
from elsc.provenance import validate_dev_selection
from elsc.resources import require_resources
from elsc.utils import atomic_json_dump, git_worktree_state, sha256_file, sha256_json


STOPWORDS_V1 = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _load_rf_metadata(
    path: str | None, dense_length: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not path or not Path(path).is_file():
        raise ValueError(f"verified receptive-field metadata is missing: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("verified") is not True:
        raise ValueError(f"RF metadata must explicitly set verified=true: {path}")
    starts = value.get("rf_start")
    ends = value.get("rf_end")
    if starts is None and isinstance(value.get("tokens"), list):
        starts = [item["start"] for item in value["tokens"]]
        ends = [item["end"] for item in value["tokens"]]
    if (
        not isinstance(starts, list)
        or not isinstance(ends, list)
        or len(starts) != dense_length
        or len(ends) != dense_length
    ):
        raise ValueError(f"RF metadata length mismatch at {path}")
    start_tensor = torch.tensor(starts, dtype=torch.float32)
    end_tensor = torch.tensor(ends, dtype=torch.float32)
    if bool((end_tensor <= start_tensor).any()):
        raise ValueError(f"RF intervals must be non-empty and half-open: {path}")
    return start_tensor, end_tensor, str(value.get("coordinate_system", "input_frame"))


@torch.no_grad()
def _teacher_clean_margin(model, tokenizer, item, negative_caption: str, config, device) -> float:
    video, _ = model.encode_video(
        item["h"].unsqueeze(0).to(device), item["valid"].unsqueeze(0).to(device)
    )
    captions = [item["caption"], negative_caption]
    encoded = [
        encode_cico_text(text, tokenizer, int(config["data"]["max_words"])) for text in captions
    ]
    ids, segments, masks = (
        torch.stack([row[index] for row in encoded]).to(device) for index in range(3)
    )
    text = model.encode_text(ids, segments, masks)
    i2t, t2i = model.bridge.score(video, text, objective=True)
    scores = model.bridge.mixed_score(i2t, t2i, float(config["model"]["dual_mix"]))[0]
    scale = model.core.clip.logit_scale.exp().detach().float()
    return float(((scores[0] - scores[1]) / scale).item())


def _add_evidence_metadata(
    final, dataset, model, tokenizer, config, device, gates, basic_clean, whitespace_clean
):
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in final:
        by_pair[record["pair_id"]].append(record)
    index_by_pair = {record.pair_id: index for index, record in enumerate(dataset.records)}
    manifest_by_pair = {record.pair_id: record for record in dataset.records}
    selected_by_video: Counter[str] = Counter()
    maximum_per_video = int(config["evidence"].get("max_pairs_per_video", 1))
    for pair_id, pair_records in by_pair.items():
        item = dataset[index_by_pair[pair_id]]
        manifest = manifest_by_pair[pair_id]
        dense_start, dense_end, coordinate = _load_rf_metadata(
            manifest.temporal_metadata, manifest.dense_length
        )
        sampled = item["dense_index"]
        valid = item["valid"]
        rf_start = dense_start.index_select(0, sampled[valid])
        rf_end = dense_end.index_select(0, sampled[valid])
        padded_start = torch.zeros_like(valid, dtype=torch.float32)
        padded_end = torch.zeros_like(valid, dtype=torch.float32)
        padded_start[valid] = rf_start
        padded_end[valid] = rf_end
        support_masks: dict[int, torch.Tensor] = {}
        physical_intervals: dict[int, tuple[float, float]] = {}
        for record in pair_records:
            support_dense = torch.tensor(record["support_dense_indices"], dtype=torch.long)
            support_start = dense_start.index_select(0, support_dense).min().item()
            support_end = dense_end.index_select(0, support_dense).max().item()
            interval = (float(support_start), float(support_end))
            physical_intervals[record["word_id"]] = interval
            support_masks[record["word_id"]] = receptive_field_closure(
                padded_start, padded_end, interval, valid
            )
        for record in pair_records:
            video_id = str(record["video_id"])
            if selected_by_video[video_id] >= maximum_per_video:
                gates["abstain_evidence_per_video_cap"] += 1
                continue
            evidence_mask = support_masks[record["word_id"]]
            forbidden = torch.zeros_like(valid)
            for other_id, other_mask in support_masks.items():
                if other_id != record["word_id"]:
                    forbidden |= other_mask
            control = select_matched_control(
                padded_start,
                padded_end,
                valid,
                evidence_mask,
                forbidden,
                seed=int(config["seed"]),
                pair_id=pair_id,
                target_id=int(record["word_id"]),
                duration_tolerance=float(config["evidence"]["control_duration_tolerance"]),
            )
            if control is None:
                gates["abstain_no_matched_control"] += 1
                continue
            # Evidence pairs whose replacement changes global BPE subsampling are excluded.
            positive_offsets = encode_with_offsets(
                item["caption"],
                tokenizer,
                int(config["data"]["max_words"]),
                basic_clean=basic_clean,
                whitespace_clean=whitespace_clean,
            )
            negative_offsets = encode_with_offsets(
                record["negative_captions"][0],
                tokenizer,
                int(config["data"]["max_words"]),
                basic_clean=basic_clean,
                whitespace_clean=whitespace_clean,
            )
            if positive_offsets.was_subsampled or negative_offsets.was_subsampled:
                gates["abstain_evidence_bpe_subsampling"] += 1
                continue
            margin = _teacher_clean_margin(
                model, tokenizer, item, record["negative_captions"][0], config, device
            )
            record["teacher_clean_margin"] = margin
            if margin < float(config["evidence"]["teacher_margin_min"]):
                gates["abstain_teacher_margin"] += 1
                continue
            record["evidence_remove_dense_indices"] = sampled[evidence_mask].tolist()
            record["control_remove_dense_indices"] = sampled[control].tolist()
            record["evidence_interval"] = list(physical_intervals[record["word_id"]])
            control_positions = control.nonzero(as_tuple=False).flatten()
            record["control_interval"] = [
                float(padded_start[control_positions].min()),
                float(padded_end[control_positions].max()),
            ]
            record["intervention_coordinate_system"] = coordinate
            record["evidence_eligible"] = True
            selected_by_video[video_id] += 1
            gates["evidence_eligible"] += 1


def mining_config_hash(config: dict[str, Any]) -> str:
    evidence = config.get("evidence", {})
    relevant = {
        "schema_version": config["schema_version"],
        "data": config["data"],
        "mining": config["mining"],
        "cache": config["cache"],
        "evidence_cache_contract": {
            key: evidence.get(key)
            for key in (
                "enabled",
                "require_verified_rf_metadata",
                "teacher_margin_min",
                "control_same_token_count",
                "control_duration_tolerance",
            )
        },
        "teacher_checkpoint": config["model"]["teacher_checkpoint"],
        "dual_mix": config["model"]["dual_mix"],
    }
    return sha256_json(relevant)


def _word_embedding(text_encoding, positions: tuple[int, ...], sample: int) -> torch.Tensor:
    index = torch.tensor(positions, device=text_encoding.tokens.device, dtype=torch.long)
    return F.normalize(text_encoding.tokens[sample].index_select(0, index).float().mean(0), dim=0)


@torch.no_grad()
def _lexical_bank(
    model,
    tokenizer,
    vocabulary: list[str],
    max_words: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows: list[torch.Tensor] = []
    for start in range(0, len(vocabulary), batch_size):
        surfaces = vocabulary[start : start + batch_size]
        encoded = [encode_cico_text(surface, tokenizer, max_words) for surface in surfaces]
        ids, segments, masks = (
            torch.stack([item[index] for item in encoded]).to(device) for index in range(3)
        )
        output = model.encode_text(ids, segments, masks)
        for index, mask in enumerate(masks):
            # A standalone lexical unit occupies every non-special position.
            positions = tuple(range(1, int(mask.sum().item()) - 1))
            if not positions:
                raise ValueError(f"word produced no BPE pieces: {surfaces[index]!r}")
            rows.append(_word_embedding(output, positions, index).cpu())
    return torch.stack(rows).numpy().astype(np.float32, copy=False)


def _record_text_units(records, tokenizer, max_words, basic_clean, whitespace_clean):
    encoded_by_pair = {}
    counts: Counter[str] = Counter()
    videos: dict[str, set[str]] = defaultdict(set)
    for record in records:
        encoded = encode_with_offsets(
            record.caption_model,
            tokenizer,
            max_words,
            basic_clean=basic_clean,
            whitespace_clean=whitespace_clean,
        )
        units = eligible_word_units(encoded, STOPWORDS_V1)
        encoded_by_pair[record.pair_id] = (encoded, units)
        for unit in units:
            counts[unit.surface] += 1
            videos[unit.surface].add(record.video_id)
    return encoded_by_pair, counts, videos


@torch.no_grad()
def build_cache(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    implementation = git_worktree_state(Path(__file__))
    resources = config.get("resources", {})
    require_resources(
        Path(config["cache"]["path"]).parent,
        device,
        min_disk_gib=float(resources.get("min_free_disk_gib", 20)),
        min_gpu_gib=float(resources.get("min_free_gpu_gib_mining", 8)),
        operation="cache mining",
    )
    data = config["data"]
    mining = config["mining"]
    cache_cfg = config["cache"]
    records = load_manifest(data["train_manifest"], expected_split="train")
    tokenizer, basic_clean, whitespace_clean, bpe_path = load_cico_tokenizer_components(config)
    teacher_path = Path(config["model"]["teacher_checkpoint"])
    validate_dev_selection(config["model"]["teacher_selection_provenance"], teacher_path)
    output = Path(cache_cfg["path"])
    existing_meta_path = output / "cache_meta.json"
    if output.exists() and any(output.iterdir()) and not existing_meta_path.is_file():
        raise CacheMismatchError(f"cache directory is non-empty but has no metadata: {output}")
    if existing_meta_path.is_file():
        existing = json.loads(existing_meta_path.read_text(encoding="utf-8"))
        validate_cache_meta(
            existing,
            {
                "config_hash": config_hash(config),
                "mining_config_hash": mining_config_hash(config),
                "manifest_hash": sha256_file(data["train_manifest"]),
                "teacher_hash": sha256_file(teacher_path),
                "tokenizer_hash": sha256_file(bpe_path),
                "language": data["caption_language"],
                "feature_fusion": f"sum:{data['alpha']}",
                "negative_source": cache_cfg.get("negatives_source", "train_visual_neighbors_v1"),
            },
        )
        artifact_hashes = {
            "records_sha256": output / "records.jsonl",
            "lexical_bank_hash": output / "lexical_bank.npy",
            "negative_table_hash": output / "negative_graph.json",
        }
        for key, path in artifact_hashes.items():
            if not path.is_file() or existing.get(key) != sha256_file(path):
                raise CacheMismatchError(f"existing cache artifact hash mismatch: {path}")
        return {**existing, "status": "reused"}
    encoded_by_pair, occurrence_counts, distinct_videos = _record_text_units(
        records, tokenizer, int(data["max_words"]), basic_clean, whitespace_clean
    )
    vocabulary = sorted(occurrence_counts)
    word_to_id = {surface: index for index, surface in enumerate(vocabulary)}
    teacher, _ = build_retriever_from_checkpoint(config, teacher_path, device=device)
    teacher.eval().requires_grad_(False)

    dataset = CiCoFeatureDataset(
        data["train_manifest"],
        feature_len=int(data["feature_len"]),
        alpha=float(data["alpha"]),
        split="train",
        include_jittered_view=True,
        seed=int(config["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["per_device_batch"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 4)),
        collate_fn=CiCoCollator(tokenizer, int(data["max_words"]), augment=False),
    )
    preliminary: list[dict[str, Any]] = []
    prototypes: list[PrototypeOccurrence] = []
    gates = Counter()
    record_by_pair = {record.pair_id: record for record in records}
    for batch in loader:
        h_a = batch["h"].to(device)
        valid_a = batch["valid"].to(device)
        h_b = batch["h_b"].to(device)
        valid_b = batch["valid_b"].to(device)
        visual_a, _ = teacher.encode_video(h_a, valid_a)
        visual_b, _ = teacher.encode_video(h_b, valid_b)
        ids, segments, mask = (tensor.to(device) for tensor in batch["clean_text"])
        text = teacher.encode_text(ids, segments, mask)
        for sample, pair_id in enumerate(batch["pair_id"]):
            manifest_record = record_by_pair[pair_id]
            encoded, units = encoded_by_pair[pair_id]
            gates["eligible_word_units"] += len(units)
            if not bool(batch["view_independent"][sample]):
                gates["abstain_non_independent_view"] += len(units)
                continue
            if not units:
                continue
            word_context = torch.stack(
                [_word_embedding(text, unit.encoded_bpe_positions, sample) for unit in units]
            )
            vis_a = visual_a.tokens[sample, 1:][valid_a[sample]]
            vis_b = visual_b.tokens[sample, 1:][valid_b[sample]]
            p_a, q_a = token_word_distributions(
                vis_a, word_context, tau_word=mining["tau_word"], tau_time=mining["tau_time"]
            )
            p_b, q_b = token_word_distributions(
                vis_b, word_context, tau_word=mining["tau_word"], tau_time=mining["tau_time"]
            )
            dense_a = batch["dense_index"][sample][batch["valid"][sample]].to(device)
            dense_b = batch["dense_index_b"][sample][batch["valid_b"][sample]].to(device)
            for word_index, unit in enumerate(units):
                support_a = select_support(
                    p_a,
                    q_a,
                    dense_a,
                    word_index,
                    mass_min=mining["mass_min"],
                    min_support_tokens=mining["min_support_tokens"],
                    max_duration_fraction=mining["max_duration_fraction"],
                    confidence_min=mining["confidence_min"],
                    second_mode_mass=mining.get("second_mode_mass", 0.30),
                )
                support_b = select_support(
                    p_b,
                    q_b,
                    dense_b,
                    word_index,
                    mass_min=mining["mass_min"],
                    min_support_tokens=mining["min_support_tokens"],
                    max_duration_fraction=mining["max_duration_fraction"],
                    confidence_min=mining["confidence_min"],
                    second_mode_mass=mining.get("second_mode_mass", 0.30),
                )
                if support_a is None or support_b is None:
                    gates["abstain_support"] += 1
                    continue
                iou, rho = reliability(
                    support_a,
                    support_b,
                    occurrence_count=occurrence_counts[unit.surface],
                    view_iou_min=mining["view_iou_min"],
                )
                if rho == 0:
                    gates["abstain_view_iou"] += 1
                    continue
                word_id = word_to_id[unit.surface]
                raw = raw_support_prototype(
                    batch["h"][sample, list(support_a.selected_positions)],
                    torch.tensor(support_a.weights),
                )
                prototypes.append(PrototypeOccurrence(word_id, manifest_record.video_id, raw))
                preliminary.append(
                    {
                        "schema_version": 1,
                        "split": "train",
                        "pair_id": pair_id,
                        "video_id": manifest_record.video_id,
                        "word_id": word_id,
                        "word_surface": unit.surface,
                        "canonical_char_span": list(unit.canonical_char_span),
                        "encoded_bpe_positions": list(unit.encoded_bpe_positions),
                        "support_dense_indices": list(support_a.dense_indices),
                        "support_weights": list(support_a.weights),
                        "support_interval": list(support_a.interval),
                        "support_interval_view_b": list(support_b.interval),
                        "coordinate_system": "dense_index",
                        "view_iou": iou,
                        "alignment_confidence": support_a.confidence,
                        "rho": rho,
                        "view_hash": batch["view_hash"][sample],
                        "caption_surfaces": [item.surface for item in encoded.words],
                        "canonical_caption": encoded.canonical,
                    }
                )
                gates["reliable_support"] += 1

    minimum_videos = int(mining["min_distinct_videos_per_word"])
    reliable_videos: dict[int, set[str]] = defaultdict(set)
    for item in prototypes:
        reliable_videos[item.word_id].add(item.video_id)
    eligible_word_ids = {
        word_id for word_id, videos in reliable_videos.items() if len(videos) >= minimum_videos
    }
    # One occurrence per video and a deterministic cap prevent long/repeated videos
    # from dominating a word prototype graph.
    capped_prototypes: list[PrototypeOccurrence] = []
    by_word_prototypes: dict[int, list[PrototypeOccurrence]] = defaultdict(list)
    for item in prototypes:
        by_word_prototypes[item.word_id].append(item)
    for word_id in sorted(by_word_prototypes):
        seen_video: set[str] = set()
        for item in sorted(by_word_prototypes[word_id], key=lambda value: value.video_id):
            if item.video_id in seen_video:
                continue
            capped_prototypes.append(item)
            seen_video.add(item.video_id)
            if len(seen_video) == int(mining["max_occurrences_per_word"]):
                break
    negative_source = cache_cfg.get("negatives_source", "train_visual_neighbors_v1")
    if negative_source == "train_visual_neighbors_v1":
        graph = build_visual_neighbor_graph(
            [item for item in capped_prototypes if item.word_id in eligible_word_ids],
            cosine_min=float(mining["visual_cosine_min"]),
            top_k=int(mining["graph_top_k"]),
            mutual=bool(mining["mutual_neighbors"]),
            device=device,
        )
    elif negative_source == "train_random_frequency_matched_v1":
        graph = build_frequency_matched_random_graph(
            {
                word_id: occurrence_counts[surface]
                for surface, word_id in word_to_id.items()
                if word_id in eligible_word_ids
            },
            seed=int(config["seed"]),
            top_k=int(mining["graph_top_k"]),
        )
    else:
        raise ValueError(f"unsupported negative source: {negative_source}")
    id_to_surface = {index: surface for surface, index in word_to_id.items()}
    final: list[dict[str, Any]] = []
    per_video = Counter()
    for record in sorted(
        preliminary,
        key=lambda value: (-value["rho"], value["pair_id"], value["canonical_char_span"][0]),
    ):
        if record["word_id"] not in eligible_word_ids:
            gates["abstain_word_video_coverage"] += 1
            continue
        if per_video[record["video_id"]] >= int(cache_cfg["max_targets_per_video"]):
            gates["abstain_target_cap"] += 1
            continue
        negatives = filter_occurrence_negatives(
            record["word_id"],
            record["word_surface"],
            set(record.pop("caption_surfaces")),
            graph,
            id_to_surface,
            max_negatives=int(cache_cfg["max_negatives_per_target"]),
        )
        if not negatives:
            gates["abstain_no_negative"] += 1
            continue
        canonical = record.pop("canonical_caption")
        record["negative_word_ids"] = negatives
        record["negative_captions"] = [
            replace_canonical_span(
                canonical, tuple(record["canonical_char_span"]), id_to_surface[value]
            )
            for value in negatives
        ]
        record["negative_source"] = negative_source
        record["evidence_eligible"] = False
        final.append(record)
        per_video[record["video_id"]] += 1
        gates["final_records"] += 1

    if config.get("evidence", {}).get("enabled"):
        _add_evidence_metadata(
            final,
            dataset,
            teacher,
            tokenizer,
            config,
            device,
            gates,
            basic_clean,
            whitespace_clean,
        )

    output.mkdir(parents=True, exist_ok=True)
    bank = _lexical_bank(
        teacher,
        tokenizer,
        vocabulary,
        int(data["max_words"]),
        device,
        int(config["train"]["per_device_batch"]),
    )
    np.save(output / "lexical_bank.npy", bank)
    vocab_records = [
        {
            "word_id": word_to_id[surface],
            "surface": surface,
            "language": data["caption_language"],
            "occurrence_count": occurrence_counts[surface],
            "distinct_video_count": len(distinct_videos[surface]),
            "bpe_ids": tokenizer.encode(surface),
            "eligible_graph": word_to_id[surface] in eligible_word_ids,
        }
        for surface in vocabulary
    ]
    atomic_json_dump(vocab_records, output / "vocab.json")
    graph_json = {
        str(word): [{"word_id": other, "score": score} for other, score in values]
        for word, values in graph.items()
    }
    atomic_json_dump(graph_json, output / "negative_graph.json")
    records_path = output / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in sorted(
            final, key=lambda value: (value["pair_id"], value["canonical_char_span"][0])
        ):
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    metadata = {
        "schema_version": 1,
        "implementation": implementation,
        "split": "train",
        "teacher_hash": sha256_file(teacher_path),
        "config_hash": config_hash(config),
        "mining_config_hash": mining_config_hash(config),
        "manifest_hash": sha256_file(data["train_manifest"]),
        "tokenizer_hash": sha256_file(bpe_path),
        "language": data["caption_language"],
        "feature_fusion": f"sum:{data['alpha']}",
        "view_sampling": "upstream_uniform+jitter1_v1",
        "mining_version": "elsc_v1",
        "stopword_version": "elsc_en_stopwords_v1",
        "stopword_hash": sha256_json(sorted(STOPWORDS_V1)),
        "negative_table_hash": sha256_file(output / "negative_graph.json"),
        "negative_source": negative_source,
        "lexical_bank_hash": sha256_file(output / "lexical_bank.npy"),
        "records_sha256": sha256_file(records_path),
        "gates": dict(gates),
        "vocabulary_size": len(vocabulary),
    }
    atomic_json_dump(metadata, output / "cache_meta.json")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train-only ELSC support and negative cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=("train",))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    config = load_config(args.config, stage="build_cache")
    result = build_cache(config, torch.device(args.device))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
