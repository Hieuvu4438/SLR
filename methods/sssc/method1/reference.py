from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .schemas import SchemaError
from .utils import sha256_file, sha256_json


REFERENCE_CACHE_SCHEMA_VERSION = 1


class ReferenceCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReferenceCacheIdentity:
    teacher_checkpoint_sha256: str
    tokenizer_sha256: str
    caption_manifest_sha256: str
    feature_manifest_sha256: str
    mixture_sampler_sha256: str
    implementation_version: str
    cache_dtype: str = "float32"

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


def freeze_reference(reference: torch.nn.Module) -> torch.nn.Module:
    reference.requires_grad_(False)
    reference.eval()
    return reference


def _require_frozen_eval(reference: torch.nn.Module) -> None:
    if reference.training or any(parameter.requires_grad for parameter in reference.parameters()):
        raise ReferenceCacheError("reference must be eval-only with every parameter frozen")


@torch.no_grad()
def encode_reference_video(
    reference: torch.nn.Module,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_frozen_eval(reference)
    if "video_features" not in batch or "video_ignore_raw" not in batch:
        raise SchemaError("reference video batch requires video_features and video_ignore_raw")
    result = reference.get_video_feat(
        batch["video_features"],
        batch["video_ignore_raw"],
        shaped=True,
        video_frame=1,
        get_hidden=True,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise ReferenceCacheError("UPRet get_video_feat must return mask, hidden tokens, class")
    video_ignore, tokens, _ = result
    if tokens.ndim != 3 or tokens.shape[1] != 65 or video_ignore.shape != tokens.shape[:2]:
        raise SchemaError("reference video encoding must be [B,65,D] with an aligned mask")
    if not bool(video_ignore[:, 0].bool().all()):
        raise SchemaError("reference video class token must be marked ignored")
    valid = ~video_ignore[:, 1:].bool()
    if not bool(valid.any(dim=-1).all()):
        raise SchemaError("reference video contains no valid local clip")
    normalized = F.normalize(tokens[:, 1:, :].float(), dim=-1, eps=1e-6)
    normalized = normalized.masked_fill(~valid[:, :, None], 0.0)
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("reference video encoding is nonfinite")
    return normalized, valid


@torch.no_grad()
def encode_reference_text(
    reference: torch.nn.Module,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    text_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_frozen_eval(reference)
    result = reference.get_text_feat(
        input_ids, token_type_ids, text_valid, shaped=False, get_hidden=True
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise ReferenceCacheError("UPRet get_text_feat must return mask, hidden tokens, class")
    returned_valid, tokens, _ = result
    if tokens.ndim != 3 or returned_valid.shape != tokens.shape[:2]:
        raise SchemaError("reference text encoding must be [B,M,D] with an aligned mask")
    if not torch.equal(returned_valid.bool(), text_valid.bool()):
        raise SchemaError("reference text mask disagrees with tokenizer mask")
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError("reference text encoding is nonfinite")
    return tokens.float(), returned_valid.bool()


def pool_reference_spans(
    reference: torch.nn.Module,
    tokens: torch.Tensor,
    spans: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Pool raw contextual tokens, then normalize one complete span per row."""
    _require_frozen_eval(reference)
    if tokens.ndim != 3 or len(spans) != tokens.shape[0]:
        raise SchemaError("tokens/spans must align as [N,M,D] and N position sequences")
    pooled: list[torch.Tensor] = []
    for row, positions in enumerate(spans):
        positions = tuple(int(position) for position in positions)
        if not positions or len(set(positions)) != len(positions):
            raise SchemaError("span positions must be non-empty and distinct")
        if min(positions) < 1 or max(positions) >= tokens.shape[1] - 1:
            raise SchemaError("span positions must be non-special in-range token indexes")
        pooled.append(tokens[row, list(positions)].float().mean(dim=0))
    output = F.normalize(torch.stack(pooled), dim=-1, eps=1e-6)
    if bool((output.norm(dim=-1) <= 1e-6).any()) or not bool(torch.isfinite(output).all()):
        raise SchemaError("reference span pooling produced a zero or nonfinite vector")
    return output


def _validate_cache_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {"reference_video_tokens", "reference_video_valid", "reference_positive_spans"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ReferenceCacheError(f"reference arrays are missing: {missing}")
    tokens = np.asarray(arrays["reference_video_tokens"])
    valid = np.asarray(arrays["reference_video_valid"])
    positives = np.asarray(arrays["reference_positive_spans"])
    if tokens.ndim != 3 or tokens.shape[1] != 64 or tokens.dtype != np.float32:
        raise ReferenceCacheError("video cache must be float32 [N,64,D]")
    if valid.shape != tokens.shape[:2] or valid.dtype != np.bool_:
        raise ReferenceCacheError("video validity cache must be bool [N,64]")
    if positives.ndim != 2 or positives.shape[1] != tokens.shape[2] or positives.dtype != np.float32:
        raise ReferenceCacheError("positive span cache must be float32 [O,D]")
    if not np.isfinite(tokens).all() or not np.isfinite(positives).all():
        raise ReferenceCacheError("reference cache contains NaN or infinity")
    if bool((np.abs(tokens[~valid]) > 0).any()):
        raise ReferenceCacheError("invalid video positions must be exactly zero-filled")
    if bool((valid.sum(axis=1) < 1).any()):
        raise ReferenceCacheError("each cached video must contain a valid clip")
    valid_norms = np.linalg.norm(tokens, axis=-1)[valid]
    if not np.allclose(valid_norms, 1.0, atol=2e-5, rtol=2e-5):
        raise ReferenceCacheError("valid reference video tokens must be normalized")
    if len(positives) and not np.allclose(
        np.linalg.norm(positives, axis=-1), 1.0, atol=2e-5, rtol=2e-5
    ):
        raise ReferenceCacheError("reference positive spans must be normalized")
    negatives = arrays.get("reference_negative_spans")
    if negatives is not None:
        negatives = np.asarray(negatives)
        if negatives.ndim != 2 or negatives.shape[1] != tokens.shape[2] or negatives.dtype != np.float32:
            raise ReferenceCacheError("negative span cache must be float32 [E,D]")


def _write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_reference_cache(
    destination: str | Path,
    *,
    identity: ReferenceCacheIdentity,
    arrays: Mapping[str, np.ndarray],
    video_index: Sequence[Mapping[str, Any]],
    positive_span_index: Sequence[Mapping[str, Any]],
    negative_span_index: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    _validate_cache_arrays(arrays)
    if len(video_index) != len(arrays["reference_video_tokens"]):
        raise ReferenceCacheError("video index length does not match video cache")
    if len(positive_span_index) != len(arrays["reference_positive_spans"]):
        raise ReferenceCacheError("positive-span index length does not match its cache")
    negatives = arrays.get("reference_negative_spans")
    if (0 if negatives is None else len(negatives)) != len(negative_span_index):
        raise ReferenceCacheError("negative-span index length does not match its cache")
    path = Path(destination)
    if path.exists():
        raise ReferenceCacheError(f"refusing to overwrite existing reference cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.building.", dir=path.parent))
    try:
        array_meta: dict[str, Any] = {}
        for name, source in arrays.items():
            source = np.asarray(source)
            array_path = staging / f"{name}.npy"
            output = np.lib.format.open_memmap(
                array_path, mode="w+", dtype=source.dtype, shape=source.shape
            )
            output[...] = source
            output.flush()
            del output
            array_meta[name] = {
                "file": array_path.name,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "sha256": sha256_file(array_path),
            }
        indexes = {
            "video_index": video_index,
            "positive_span_index": positive_span_index,
            "negative_span_index": negative_span_index,
        }
        index_meta: dict[str, Any] = {}
        for name, records in indexes.items():
            index_path = staging / f"{name}.jsonl"
            _write_jsonl(records, index_path)
            index_meta[name] = {
                "file": index_path.name,
                "count": len(records),
                "sha256": sha256_file(index_path),
            }
        metadata = {
            "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
            "status": "complete",
            "identity": asdict(identity),
            "identity_sha256": identity.digest,
            "arrays": array_meta,
            "indexes": index_meta,
        }
        metadata["content_sha256"] = sha256_json(metadata)
        meta_path = staging / "cache_meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_reference_cache(
    directory: str | Path,
    *,
    expected_identity: ReferenceCacheIdentity,
) -> dict[str, np.memmap]:
    root = Path(directory)
    meta_path = root / "cache_meta.json"
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceCacheError(f"cannot read completed reference cache: {error}") from error
    if metadata.get("status") != "complete" or metadata.get("identity_sha256") != expected_identity.digest:
        raise ReferenceCacheError("reference cache identity differs from the requested teacher/data/code")
    arrays: dict[str, np.memmap] = {}
    for name, description in metadata.get("arrays", {}).items():
        source = root / description["file"]
        if sha256_file(source) != description["sha256"]:
            raise ReferenceCacheError(f"reference cache hash mismatch: {name}")
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != description["shape"] or str(array.dtype) != description["dtype"]:
            raise ReferenceCacheError(f"reference cache shape/dtype mismatch: {name}")
        arrays[name] = array
    _validate_cache_arrays(arrays)
    for description in metadata.get("indexes", {}).values():
        source = root / description["file"]
        if sha256_file(source) != description["sha256"]:
            raise ReferenceCacheError(f"reference index hash mismatch: {source.name}")
    return arrays


def write_negative_span_cache(
    cache_root: str | Path,
    *,
    reference_identity_sha256: str,
    mining_content_sha256: str,
    vectors: np.ndarray,
    edit_uids: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(vectors)
    if values.ndim != 2 or values.dtype != np.float32 or len(values) != len(edit_uids):
        raise ReferenceCacheError("negative spans must be float32 [E,D] aligned with edit IDs")
    if not np.isfinite(values).all() or (
        len(values)
        and not np.allclose(np.linalg.norm(values, axis=-1), 1.0, atol=2e-5, rtol=2e-5)
    ):
        raise ReferenceCacheError("negative span vectors must be finite and normalized")
    if len(set(edit_uids)) != len(edit_uids):
        raise ReferenceCacheError("negative span edit IDs must be unique")
    destination = Path(cache_root) / "negative_spans"
    if destination.exists():
        raise ReferenceCacheError(f"refusing to overwrite negative span cache: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building.", dir=destination.parent)
    )
    try:
        array_path = staging / "reference_negative_spans.npy"
        output = np.lib.format.open_memmap(
            array_path, mode="w+", dtype=np.float32, shape=values.shape
        )
        output[...] = values
        output.flush()
        del output
        index_path = staging / "negative_span_index.jsonl"
        _write_jsonl(
            [{"row": row, "edit_uid": edit_uid} for row, edit_uid in enumerate(edit_uids)],
            index_path,
        )
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "reference_identity_sha256": reference_identity_sha256,
            "mining_content_sha256": mining_content_sha256,
            "count": len(edit_uids),
            "dimension": values.shape[1],
            "array_sha256": sha256_file(array_path),
            "index_sha256": sha256_file(index_path),
        }
        metadata["content_sha256"] = sha256_json(metadata)
        (staging / "negative_cache_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_negative_span_cache(
    cache_root: str | Path,
    *,
    reference_identity_sha256: str,
    mining_content_sha256: str,
) -> tuple[np.memmap, dict[str, int]]:
    root = Path(cache_root) / "negative_spans"
    try:
        metadata = json.loads((root / "negative_cache_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceCacheError(f"cannot read completed negative span cache: {error}") from error
    if (
        metadata.get("status") != "complete"
        or metadata.get("reference_identity_sha256") != reference_identity_sha256
        or metadata.get("mining_content_sha256") != mining_content_sha256
    ):
        raise ReferenceCacheError("negative span cache identity mismatch")
    array_path = root / "reference_negative_spans.npy"
    index_path = root / "negative_span_index.jsonl"
    if sha256_file(array_path) != metadata.get("array_sha256") or sha256_file(
        index_path
    ) != metadata.get("index_sha256"):
        raise ReferenceCacheError("negative span cache file hash mismatch")
    vectors = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (metadata.get("count"), metadata.get("dimension")):
        raise ReferenceCacheError("negative span cache shape mismatch")
    index: dict[str, int] = {}
    rows: set[int] = set()
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["edit_uid"] in index:
                raise ReferenceCacheError("duplicate edit ID in negative span index")
            row = int(record["row"])
            index[record["edit_uid"]] = row
            rows.add(row)
    if len(index) != len(vectors) or rows != set(range(len(vectors))):
        raise ReferenceCacheError("negative span index count mismatch")
    return vectors, index
