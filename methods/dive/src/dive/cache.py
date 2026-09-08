from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


class CacheError(ValueError):
    """A tensor cache is stale, incomplete, corrupt, or violates its schema."""


_FINGERPRINT_SCHEMA = "dive_cache_fingerprint.v1"
_INDEX_SCHEMA = "dive_tensor_cache_index.v2"
_SHARD_SCHEMA = "dive_tensor_cache_shard.v2"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_COMPONENTS: dict[str, frozenset[str]] = {
    "rgb_local": frozenset(
        {
            "dataset_content",
            "extractor_state",
            "frame_map",
            "grid_view",
            "preprocess",
            "stage",
            "dtype",
        }
    ),
    "native_baseline": frozenset(
        {
            "baseline_state",
            "preprocessing",
            "tokenizer",
            "sequence_selection",
            "masks",
            "sample_ids",
        }
    ),
    "text_units": frozenset(
        {
            "text_encoder_state",
            "text_model",
            "token_ids",
            "unit_mapping",
            "normalization",
            "epsilon",
            "dtype",
        }
    ),
    "reference_local": frozenset(
        {
            "reference_state",
            "rgb_preprocess",
            "pose_preprocess",
            "grid_view",
            "bn_policy",
            "dtype",
        }
    ),
    "pair_scores": frozenset(
        {
            "feature_hashes",
            "scorer_version",
            "tau_alignment",
            "native_mixing",
            "score_scale",
        }
    ),
    "contrast_bank": frozenset(
        {
            "train_manifest",
            "baseline",
            "reference",
            "schema_audit",
            "units",
            "mining_config",
            "support_config",
            "timestamps",
        }
    ),
    "student_gallery": frozenset(
        {
            "student_checkpoint",
            "reference",
            "baseline",
            "text_contracts",
            "grid",
            "dtype",
        }
    ),
}
_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def _canonicalize(value: Any, path: str = "components") -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CacheError(f"{path} keys must be nonempty strings")
            normalized[key] = _canonicalize(item, f"{path}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item, f"{path}[]") for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CacheError(f"{path} contains a nonfinite number")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise CacheError(f"{path} contains unsupported fingerprint value {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class CacheFingerprint:
    artifact: str
    components: Mapping[str, Any]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _FINGERPRINT_SCHEMA,
            "artifact": self.artifact,
            "components": _canonicalize(self.components),
            "sha256": self.digest,
        }


def make_cache_fingerprint(artifact: str, components: Mapping[str, Any]) -> CacheFingerprint:
    if artifact not in _REQUIRED_COMPONENTS:
        raise CacheError(f"unsupported cache artifact type: {artifact}")
    canonical_components = _canonicalize(components)
    missing = _REQUIRED_COMPONENTS[artifact] - set(canonical_components)
    if missing:
        raise CacheError(f"{artifact} fingerprint is missing components: {sorted(missing)}")
    empty = [key for key, value in canonical_components.items() if value in (None, "", [], {})]
    if empty:
        raise CacheError(f"{artifact} fingerprint has empty components: {sorted(empty)}")
    envelope = {
        "schema_version": _FINGERPRINT_SCHEMA,
        "artifact": artifact,
        "components": canonical_components,
    }
    digest = hashlib.sha256(_canonical_json(envelope).encode("utf-8")).hexdigest()
    if artifact == "reference_local" and canonical_components["dtype"] != "float32":
        raise CacheError("correctness reference cache must use float32")
    return CacheFingerprint(artifact=artifact, components=canonical_components, digest=digest)


def _validate_fingerprint(fingerprint: CacheFingerprint) -> None:
    rebuilt = make_cache_fingerprint(fingerprint.artifact, fingerprint.components)
    if fingerprint.digest != rebuilt.digest or fingerprint.to_dict() != rebuilt.to_dict():
        raise CacheError("cache fingerprint record is internally inconsistent")


@dataclass(frozen=True)
class TensorShard:
    shard_id: str
    ordered_ids: tuple[str, ...]
    tensors: Mapping[str, Tensor]
    masks: Mapping[str, Tensor]
    timestamps: Mapping[str, Tensor]
    metadata: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checksum(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value if _SHA256.fullmatch(value) else None


def _validate_named_tensors(
    values: Mapping[str, Tensor],
    *,
    kind: str,
    count: int,
    expected_dtype: torch.dtype | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping):
        raise CacheError(f"{kind} must be a mapping")
    specs: dict[str, dict[str, Any]] = {}
    for name, tensor in values.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, Tensor):
            raise CacheError(f"{kind} must map nonempty names to tensors")
        if tensor.ndim == 0 or tensor.shape[0] != count:
            raise CacheError(f"{kind}.{name} must have ordered ID count as its first dimension")
        if kind == "masks" and tensor.dtype != torch.bool:
            raise CacheError(f"masks.{name} must be bool with True=valid")
        if kind == "tensors" and tensor.is_floating_point():
            if expected_dtype is not None and tensor.dtype != expected_dtype:
                raise CacheError(
                    f"tensors.{name} dtype {tensor.dtype} differs from fingerprint dtype {expected_dtype}"
                )
            if not torch.isfinite(tensor).all():
                raise CacheError(f"tensors.{name} contains NaN/Inf")
        if kind == "timestamps":
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise CacheError(f"timestamps.{name} must be finite floating point")
        specs[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    return specs


def _validate_shard(
    shard: TensorShard, fingerprint: CacheFingerprint
) -> dict[str, dict[str, dict[str, Any]]]:
    if not _SAFE_NAME.fullmatch(shard.shard_id):
        raise CacheError(f"unsafe shard ID: {shard.shard_id!r}")
    if not shard.ordered_ids or any(
        not isinstance(item, str) or not item for item in shard.ordered_ids
    ):
        raise CacheError("tensor shard ordered IDs must be nonempty strings")
    if len(shard.ordered_ids) != len(set(shard.ordered_ids)):
        raise CacheError(f"tensor shard {shard.shard_id} contains duplicate IDs")
    if not shard.tensors:
        raise CacheError(f"tensor shard {shard.shard_id} contains no feature tensors")
    if not isinstance(shard.metadata, Mapping):
        raise CacheError("tensor shard metadata must be a mapping")
    _canonicalize(shard.metadata, "metadata")
    dtype_name = fingerprint.components.get("dtype")
    expected_dtype = _DTYPES.get(str(dtype_name)) if dtype_name is not None else None
    if dtype_name is not None and expected_dtype is None:
        raise CacheError(f"unsupported fingerprint dtype: {dtype_name}")
    count = len(shard.ordered_ids)
    return {
        "tensors": _validate_named_tensors(
            shard.tensors, kind="tensors", count=count, expected_dtype=expected_dtype
        ),
        "masks": _validate_named_tensors(
            shard.masks, kind="masks", count=count, expected_dtype=None
        ),
        "timestamps": _validate_named_tensors(
            shard.timestamps, kind="timestamps", count=count, expected_dtype=None
        ),
    }


def _cpu_contiguous(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().contiguous() for name, tensor in values.items()}


class TensorCacheWriter:
    """One-pass writer that can validate and resume atomic unpublished shards."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        namespace: str,
        fingerprint: CacheFingerprint,
        resume: bool = False,
    ) -> None:
        if not _SAFE_NAME.fullmatch(namespace):
            raise CacheError(f"unsafe cache namespace: {namespace!r}")
        _validate_fingerprint(fingerprint)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.fingerprint = fingerprint
        self._index_path = self.output / "index.json"
        self._index_shards: list[dict[str, Any]] = []
        self._shard_ids: set[str] = set()
        self._all_ids: list[str] = []
        self._seen_ids: set[str] = set()
        self._finalized = False
        self._reopened_published = False
        existing = sorted(self.output.glob("shard-*.pt"))
        if self._index_path.exists() and not resume:
            raise CacheError(f"tensor cache is already published: {self.output}")
        if existing and not resume:
            raise CacheError(f"tensor cache contains unpublished shards: {self.output}")
        if resume:
            for path in existing:
                self._adopt(path)
            if self._index_path.exists():
                self._validate_published_index()
                self._finalized = True
                self._reopened_published = True

    @property
    def completed_shard_ids(self) -> frozenset[str]:
        return frozenset(self._shard_ids)

    def ordered_ids_for(self, shard_id: str) -> tuple[str, ...]:
        for descriptor in self._index_shards:
            if descriptor["shard_id"] == shard_id:
                return tuple(descriptor["ordered_ids"])
        raise CacheError(f"tensor cache has no completed shard {shard_id}")

    def _check_new(self, shard: TensorShard) -> dict[str, dict[str, dict[str, Any]]]:
        if shard.shard_id in self._shard_ids:
            raise CacheError("tensor cache shard IDs must be unique")
        repeated = sorted(set(shard.ordered_ids) & self._seen_ids)
        if repeated:
            raise CacheError(f"tensor cache sample IDs must be unique across shards: {repeated}")
        return _validate_shard(shard, self.fingerprint)

    def _record(
        self,
        shard: TensorShard,
        *,
        filename: str,
        checksum: str,
        specs: Mapping[str, Any],
    ) -> None:
        self._shard_ids.add(shard.shard_id)
        self._seen_ids.update(shard.ordered_ids)
        self._all_ids.extend(shard.ordered_ids)
        self._index_shards.append(
            {
                "shard_id": shard.shard_id,
                "filename": filename,
                "record_count": len(shard.ordered_ids),
                "ordered_ids": list(shard.ordered_ids),
                "ordered_ids_sha256": hashlib.sha256(
                    _canonical_json(list(shard.ordered_ids)).encode("utf-8")
                ).hexdigest(),
                "sha256": checksum,
                "specs": specs,
                "complete": True,
            }
        )

    def _adopt(self, path: Path) -> None:
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        expected_checksum = _read_checksum(checksum_path)
        if expected_checksum is None:
            raise CacheError(f"unpublished tensor shard checksum is missing: {path.name}")
        actual_checksum = _sha256(path)
        if actual_checksum != expected_checksum:
            raise CacheError(f"unpublished tensor shard checksum mismatch: {path.name}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CacheError(f"unpublished tensor shard cannot be resumed: {path.name}") from exc
        shard_id = str(payload.get("shard_id", "")) if isinstance(payload, Mapping) else ""
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != _SHARD_SCHEMA
            or payload.get("namespace") != self.namespace
            or payload.get("fingerprint_sha256") != self.fingerprint.digest
            or path.name != f"shard-{shard_id}.pt"
        ):
            raise CacheError(f"unpublished tensor shard provenance is invalid: {path.name}")
        try:
            shard = TensorShard(
                shard_id=shard_id,
                ordered_ids=tuple(payload["ordered_ids"]),
                tensors=payload["tensors"],
                masks=payload["masks"],
                timestamps=payload["timestamps"],
                metadata=payload["metadata"],
            )
        except KeyError as exc:
            raise CacheError(f"unpublished tensor shard is missing {exc.args[0]}") from exc
        specs = self._check_new(shard)
        self._record(shard, filename=path.name, checksum=actual_checksum, specs=specs)

    def _index_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _INDEX_SCHEMA,
            "namespace": self.namespace,
            "fingerprint": self.fingerprint.to_dict(),
            "record_count": len(self._all_ids),
            "ordered_ids_sha256": hashlib.sha256(
                _canonical_json(self._all_ids).encode("utf-8")
            ).hexdigest(),
            "shards": [
                {key: value for key, value in item.items() if key != "ordered_ids"}
                for item in self._index_shards
            ],
        }

    def _validate_published_index(self) -> None:
        try:
            value = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError("published tensor cache index is unreadable") from exc
        if value != self._index_payload():
            raise CacheError("CACHE_HASH_MISMATCH: published tensor cache index differs")

    def add(self, shard: TensorShard) -> None:
        if self._finalized:
            raise CacheError("cannot append a shard after cache finalization")
        specs = self._check_new(shard)
        filename = f"shard-{shard.shard_id}.pt"
        destination = self.output / filename
        if destination.exists():
            raise CacheError(f"tensor cache shard already exists: {filename}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = {
            "schema_version": _SHARD_SCHEMA,
            "namespace": self.namespace,
            "fingerprint_sha256": self.fingerprint.digest,
            "shard_id": shard.shard_id,
            "ordered_ids": list(shard.ordered_ids),
            "tensors": _cpu_contiguous(shard.tensors),
            "masks": _cpu_contiguous(shard.masks),
            "timestamps": _cpu_contiguous(shard.timestamps),
            "metadata": _canonicalize(shard.metadata, "metadata"),
        }
        torch.save(payload, temporary)
        checksum = _sha256(temporary)
        checksum_path = destination.with_suffix(destination.suffix + ".sha256")
        checksum_tmp = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
        checksum_tmp.write_text(checksum + "\n", encoding="ascii")
        os.replace(checksum_tmp, checksum_path)
        os.replace(temporary, destination)
        self._record(shard, filename=filename, checksum=checksum, specs=specs)

    def finalize(self) -> Path:
        if self._finalized:
            if self._reopened_published:
                return self._index_path
            raise CacheError("tensor cache writer was already finalized")
        if not self._index_shards:
            raise CacheError("tensor cache requires at least one shard")
        index_tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        index_tmp.write_text(
            json.dumps(self._index_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(index_tmp, self._index_path)
        self._finalized = True
        return self._index_path


def write_tensor_cache(
    output_dir: str | Path,
    *,
    namespace: str,
    fingerprint: CacheFingerprint,
    shards: Iterable[TensorShard],
) -> Path:
    """Atomically write checksummed shards, publishing the index only after completion."""
    writer = TensorCacheWriter(
        output_dir,
        namespace=namespace,
        fingerprint=fingerprint,
    )
    for shard in shards:
        writer.add(shard)
    return writer.finalize()


def load_tensor_cache(
    output_dir: str | Path,
    *,
    expected_namespace: str,
    expected_fingerprint: CacheFingerprint,
    expected_ordered_ids: Sequence[str] | None = None,
) -> tuple[TensorShard, ...]:
    """Validate a complete cache and return tensors with explicit ID ordering."""
    _validate_fingerprint(expected_fingerprint)
    output = Path(output_dir)
    index_path = output / "index.json"
    if not index_path.is_file():
        raise CacheError(f"incomplete tensor cache (index missing): {output}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheError("tensor cache index is unreadable") from exc
    if index.get("schema_version") != _INDEX_SCHEMA:
        raise CacheError("unsupported tensor cache index schema")
    if index.get("namespace") != expected_namespace:
        raise CacheError("CACHE_NAMESPACE_MISMATCH: tensor cache split/namespace differs")
    if index.get("fingerprint") != expected_fingerprint.to_dict():
        raise CacheError("CACHE_HASH_MISMATCH: full tensor cache fingerprint differs")
    indexed_shards = index.get("shards")
    if not isinstance(indexed_shards, list) or not indexed_shards:
        raise CacheError("tensor cache index has no shards")
    loaded: list[TensorShard] = []
    all_ids: list[str] = []
    for descriptor in indexed_shards:
        if not isinstance(descriptor, Mapping):
            raise CacheError("tensor cache shard descriptor must be a mapping")
        if descriptor.get("complete") is not True:
            raise CacheError("tensor cache index references an incomplete shard")
        filename = descriptor.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CacheError("tensor cache index contains an unsafe shard filename")
        path = output / filename
        if not path.is_file() or _sha256(path) != descriptor.get("sha256"):
            raise CacheError(f"tensor cache shard checksum mismatch: {filename}")
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if _read_checksum(sidecar) != descriptor.get("sha256"):
            raise CacheError(f"tensor cache shard checksum sidecar mismatch: {filename}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CacheError(f"tensor cache shard cannot be deserialized: {filename}") from exc
        if not isinstance(payload, Mapping):
            raise CacheError("tensor cache shard payload must be a mapping")
        if payload.get("schema_version") != _SHARD_SCHEMA:
            raise CacheError("unsupported tensor cache shard schema")
        if payload.get("namespace") != expected_namespace:
            raise CacheError("tensor cache shard namespace differs")
        if payload.get("fingerprint_sha256") != expected_fingerprint.digest:
            raise CacheError("tensor cache shard fingerprint differs")
        if payload.get("shard_id") != descriptor.get("shard_id"):
            raise CacheError("tensor cache shard ID differs from index")
        ordered_ids = tuple(payload.get("ordered_ids", ()))
        count = int(descriptor.get("record_count", -1))
        if len(ordered_ids) != count:
            raise CacheError("tensor cache shard record count differs")
        ids_hash = hashlib.sha256(_canonical_json(list(ordered_ids)).encode("utf-8")).hexdigest()
        if ids_hash != descriptor.get("ordered_ids_sha256"):
            raise CacheError("tensor cache shard ordered IDs differ from index")
        try:
            shard = TensorShard(
                shard_id=str(payload["shard_id"]),
                ordered_ids=ordered_ids,
                tensors=payload["tensors"],
                masks=payload["masks"],
                timestamps=payload["timestamps"],
                metadata=payload["metadata"],
            )
        except KeyError as exc:
            raise CacheError(f"tensor cache shard is missing field: {exc.args[0]}") from exc
        actual_specs = _validate_shard(shard, expected_fingerprint)
        if actual_specs != descriptor.get("specs"):
            raise CacheError("tensor cache shard shape/dtype specs differ from index")
        loaded.append(shard)
        all_ids.extend(ordered_ids)
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != index.get("record_count"):
        raise CacheError("tensor cache contains duplicate IDs or wrong total count")
    all_ids_hash = hashlib.sha256(_canonical_json(all_ids).encode("utf-8")).hexdigest()
    if all_ids_hash != index.get("ordered_ids_sha256"):
        raise CacheError("tensor cache global ID order differs from index")
    if expected_ordered_ids is not None and tuple(all_ids) != tuple(expected_ordered_ids):
        raise CacheError("CACHE_ORDER_MISMATCH: cache ID order differs from requested manifest")
    return tuple(loaded)
