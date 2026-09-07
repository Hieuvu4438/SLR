from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from elsc.utils import stable_seed
from elsc.utils import sha256_file


class CacheMismatchError(ValueError):
    pass


REQUIRED_CACHE_META = {
    "schema_version",
    "split",
    "teacher_hash",
    "config_hash",
    "manifest_hash",
    "tokenizer_hash",
    "language",
    "feature_fusion",
    "view_sampling",
    "mining_version",
    "negative_table_hash",
    "lexical_bank_hash",
}


def validate_cache_meta(meta: dict[str, Any], expected: dict[str, Any]) -> None:
    missing = REQUIRED_CACHE_META - set(meta)
    if missing:
        raise CacheMismatchError(f"cache metadata missing: {sorted(missing)}")
    if meta["schema_version"] != 1 or meta["split"] != "train":
        raise CacheMismatchError("auxiliary cache must be schema v1 and train-only")
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if value is not None and meta.get(key) != value
    }
    if mismatches:
        details = ", ".join(f"{key}: {old!r} != {new!r}" for key, (old, new) in mismatches.items())
        raise CacheMismatchError(f"cache metadata mismatch: {details}")


def _shuffled_lexical_id(word_id: int, vocabulary_size: int, seed: int) -> int:
    if vocabulary_size < 2:
        raise CacheMismatchError("shuffled lexical control requires at least two words")
    if not 0 <= word_id < vocabulary_size:
        raise CacheMismatchError(f"word ID {word_id} is outside lexical bank")
    offset = 1 + stable_seed(seed, "shuffled_lexical_vocab_v1") % (vocabulary_size - 1)
    return (word_id + offset) % vocabulary_size


class AuxiliaryCache:
    def __init__(self, directory: str | Path, expected_meta: dict[str, Any]):
        root = Path(directory)
        with (root / "cache_meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        validate_cache_meta(self.meta, expected_meta)
        self.records_path = root / "records.jsonl"
        self.by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                if record.get("split") != "train":
                    raise CacheMismatchError(f"records.jsonl:{line_number}: non-train record")
                self.by_pair[str(record["pair_id"])].append(record)
        declared = self.meta.get("records_sha256")
        if declared and declared != sha256_file(self.records_path):
            raise CacheMismatchError("records.jsonl hash does not match cache metadata")

    def lookup(self, pair_ids: Iterable[str], view_hashes: Iterable[str]) -> list[list[dict[str, Any]]]:
        result: list[list[dict[str, Any]]] = []
        for pair_id, view_hash in zip(pair_ids, view_hashes, strict=True):
            records = self.by_pair.get(str(pair_id), [])
            for record in records:
                if record.get("view_hash") != view_hash:
                    raise CacheMismatchError(
                        f"canonical view mismatch for pair {pair_id}: cache={record.get('view_hash')} batch={view_hash}"
                    )
            result.append(records)
        return result


def lexical_tensors_from_batch(
    z: torch.Tensor,
    dense_index: torch.Tensor,
    records_by_sample: list[list[dict[str, Any]]],
    lexical_bank: torch.Tensor,
    *,
    support_mode: str = "teacher",
    seed: int = 42,
    random_span_duration_tolerance: float = 0.10,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Flatten ragged cache records into occurrence tensors without position assumptions."""
    occurrences: list[tuple[torch.Tensor, torch.Tensor, int, list[int], float]] = []
    for sample_index, records in enumerate(records_by_sample):
        position_by_dense = {
            int(value): position for position, value in enumerate(dense_index[sample_index].tolist()) if value >= 0
        }
        for record in records:
            support_ids = [int(value) for value in record["support_dense_indices"]]
            try:
                positions = [position_by_dense[value] for value in support_ids]
            except KeyError as error:
                raise CacheMismatchError(
                    f"support dense index {error.args[0]} absent from canonical view for {record['pair_id']}"
                ) from error
            weights = torch.tensor(record["support_weights"], device=z.device, dtype=torch.float32)
            if len(weights) != len(positions):
                raise CacheMismatchError("support index/weight lengths differ")
            if support_mode == "random_matched":
                if not 0.0 <= random_span_duration_tolerance <= 1.0:
                    raise ValueError("random span duration tolerance must be in [0,1]")
                dense_values = dense_index[sample_index].tolist()
                valid_positions = [
                    position for position, value in enumerate(dense_values) if value >= 0
                ]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(stable_seed(seed, record["pair_id"], record["word_id"], "random_matched"))
                width = len(positions)
                teacher_positions = set(positions)
                support_duration = (
                    int(dense_values[positions[-1]]) - int(dense_values[positions[0]]) + 1
                )
                candidates = [
                    valid_positions[start : start + width]
                    for start in range(len(valid_positions) - width + 1)
                    if not teacher_positions.intersection(
                        valid_positions[start : start + width]
                    )
                    and abs(
                        (
                            int(dense_values[valid_positions[start + width - 1]])
                            - int(dense_values[valid_positions[start]])
                            + 1
                        )
                        - support_duration
                    )
                    <= random_span_duration_tolerance * support_duration
                ]
                if not candidates:
                    continue
                positions = candidates[int(torch.randint(len(candidates), (), generator=generator).item())]
            elif support_mode not in {"teacher", "shuffled_lexical"}:
                raise ValueError(f"unsupported support mode: {support_mode}")
            word_id = int(record["word_id"])
            negatives = [int(value) for value in record.get("negative_word_ids", [])]
            if not negatives:
                continue
            occurrences.append((z[sample_index, positions], weights, word_id, negatives, float(record["rho"])))
    if not occurrences:
        empty_tokens = z[:0, :0, :]
        return (
            empty_tokens,
            z[:0, :0, 0],
            z[:0, 0, :],
            empty_tokens,
            torch.empty((0, 0), dtype=torch.bool, device=z.device),
            z[:0, 0, 0],
            torch.empty((0,), dtype=torch.long, device=z.device),
        )
    max_support = max(item[0].shape[0] for item in occurrences)
    max_negatives = max(len(item[3]) for item in occurrences)
    count = len(occurrences)
    dimension = z.shape[-1]
    support_z = z.new_zeros((count, max_support, dimension))
    support_weights = z.new_zeros((count, max_support), dtype=torch.float32)
    positive = z.new_zeros((count, dimension), dtype=torch.float32)
    negative = z.new_zeros((count, max_negatives, dimension), dtype=torch.float32)
    negative_valid = torch.zeros((count, max_negatives), dtype=torch.bool, device=z.device)
    rho = z.new_zeros((count,), dtype=torch.float32)
    word_ids = torch.empty((count,), dtype=torch.long, device=z.device)
    for index, (tokens, weights, word_id, negatives, reliability) in enumerate(occurrences):
        if support_mode == "shuffled_lexical":
            word_id = _shuffled_lexical_id(word_id, len(lexical_bank), seed)
            negatives = [
                _shuffled_lexical_id(value, len(lexical_bank), seed)
                for value in negatives
            ]
        support_z[index, : len(tokens)] = tokens
        support_weights[index, : len(weights)] = weights
        positive[index] = lexical_bank[word_id].to(z.device)
        negative[index, : len(negatives)] = lexical_bank[negatives].to(z.device)
        negative_valid[index, : len(negatives)] = True
        rho[index] = reliability
        word_ids[index] = word_id
    return support_z, support_weights, positive, negative, negative_valid, rho, word_ids
