from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .config import config_hash


class ArtifactError(ValueError):
    """A run artifact is missing, stale, corrupt, or outside its owned run directory."""


ArtifactScope = Literal["shared", "experiment"]
_STATE_SCHEMA = "dive_run_state.v1"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError("artifact metadata must be finite JSON data") from exc


def _sha256_file(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size, 1


def _sha256_directory(path: Path) -> tuple[str, int, int]:
    entries: list[dict[str, Any]] = []
    size = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise ArtifactError(f"artifact directories cannot contain symlinks: {child}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise ArtifactError(f"artifact directories cannot contain special files: {child}")
        checksum, file_size, _ = _sha256_file(child)
        size += file_size
        entries.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": checksum,
                "size_bytes": file_size,
            }
        )
    digest = hashlib.sha256(_canonical_json(entries).encode("utf-8")).hexdigest()
    return digest, size, len(entries)


def hash_artifact(path: str | Path) -> tuple[str, int, int, str]:
    """Hash one regular file or a deterministic, symlink-free directory tree."""
    source = Path(path)
    if source.is_symlink():
        raise ArtifactError(f"artifact cannot be a symlink: {source}")
    if source.is_file():
        digest, size, count = _sha256_file(source)
        return digest, size, count, "file"
    if source.is_dir():
        digest, size, count = _sha256_directory(source)
        return digest, size, count, "directory"
    raise ArtifactError(f"artifact does not exist or is not a regular file/directory: {source}")


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    path: str
    kind: str
    sha256: str
    size_bytes: int
    file_count: int


@dataclass(frozen=True)
class ResolvedArtifact:
    stage: str
    name: str
    scope: ArtifactScope
    path: Path
    record: ArtifactRecord


class ArtifactResolver:
    """Own run layout and resolve only checksummed outputs registered by parent stages."""

    def __init__(self, config: Mapping[str, Any], *, output_root: str | Path | None = None) -> None:
        run = config.get("run")
        if not isinstance(run, Mapping):
            raise ArtifactError("run config must be a mapping")
        seed = run.get("seed")
        comparison_group = run.get("comparison_group")
        variant = run.get("variant")
        configured_root = run.get("output_root")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ArtifactError("run.seed must be a nonnegative integer")
        for field, value in (("comparison_group", comparison_group), ("variant", variant)):
            if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
                raise ArtifactError(f"run.{field} is not a safe path component")
        root_value = configured_root if output_root is None else output_root
        if not isinstance(root_value, (str, Path)) or not str(root_value):
            raise ArtifactError("run.output_root must be a nonempty path")
        self.config = config
        self.config_sha256 = config_hash(config)
        self.output_root = Path(root_value).expanduser().resolve()
        self.seed = seed
        self.comparison_group = comparison_group
        self.variant = variant
        self.shared_root = self.output_root / "shared" / f"seed{seed}"
        self.experiment_root = self.output_root / comparison_group / variant / f"seed{seed}"

    def root(self, scope: ArtifactScope) -> Path:
        if scope == "shared":
            return self.shared_root
        if scope == "experiment":
            return self.experiment_root
        raise ArtifactError(f"unknown artifact scope: {scope}")

    def state_path(self, scope: ArtifactScope) -> Path:
        return self.root(scope) / "run_state.json"

    def output_path(self, scope: ArtifactScope, *parts: str) -> Path:
        if not parts or any(not _SAFE_NAME.fullmatch(part) for part in parts):
            raise ArtifactError("artifact output path components must be safe nonempty names")
        return self.root(scope).joinpath(*parts)

    def owned_path(self, scope: ArtifactScope, path: str | Path) -> Path:
        """Return a canonical path only when it remains inside the selected run scope."""
        return self._relative_owned_path(scope, path)[0]

    def _empty_state(self, scope: ArtifactScope) -> dict[str, Any]:
        layout: dict[str, Any] = {
            "scope": scope,
            "output_root": str(self.output_root),
            "seed": self.seed,
        }
        if scope == "experiment":
            layout.update(
                {
                    "comparison_group": self.comparison_group,
                    "variant": self.variant,
                }
            )
        return {
            "schema_version": _STATE_SCHEMA,
            "layout": layout,
            "stages": {},
        }

    def _load_state(self, scope: ArtifactScope) -> dict[str, Any]:
        path = self.state_path(scope)
        if not path.exists():
            return self._empty_state(scope)
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"run state is not a regular file: {path}")
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"run state is unreadable: {path}") from exc
        expected = self._empty_state(scope)
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != _STATE_SCHEMA
            or state.get("layout") != expected["layout"]
            or not isinstance(state.get("stages"), dict)
        ):
            raise ArtifactError(f"run state layout/schema mismatch: {path}")
        return state

    def _relative_owned_path(self, scope: ArtifactScope, path: str | Path) -> tuple[Path, str]:
        source = Path(path).expanduser()
        if source.is_symlink():
            raise ArtifactError(f"artifact cannot be a symlink: {source}")
        resolved = source.resolve()
        root = self.root(scope).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ArtifactError(f"artifact path escapes {scope} run root: {resolved}") from exc
        if relative == "run_state.json" or relative.endswith("/run_state.json"):
            raise ArtifactError("run_state.json cannot register itself as an output")
        return resolved, relative

    def _make_record(
        self, scope: ArtifactScope, stage: str, name: str, path: str | Path
    ) -> ArtifactRecord:
        resolved, relative = self._relative_owned_path(scope, path)
        digest, size, file_count, kind = hash_artifact(resolved)
        identity = {
            "scope": scope,
            "stage": stage,
            "name": name,
            "path": relative,
            "sha256": digest,
        }
        artifact_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        return ArtifactRecord(
            artifact_id=artifact_id,
            path=relative,
            kind=kind,
            sha256=digest,
            size_bytes=size,
            file_count=file_count,
        )

    def record_stage(
        self,
        stage: str,
        outputs: Mapping[str, str | Path],
        *,
        scope: ArtifactScope,
        parents: Sequence[ResolvedArtifact] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, ArtifactRecord]:
        """Atomically register an immutable stage result after validating all inputs/outputs."""
        if not _SAFE_NAME.fullmatch(stage):
            raise ArtifactError("stage must be a safe nonempty name")
        if not outputs:
            raise ArtifactError("a completed stage must register at least one output")
        if any(not _SAFE_NAME.fullmatch(name) for name in outputs):
            raise ArtifactError("artifact output names must be safe nonempty names")
        for parent in parents:
            current = self.resolve(parent.stage, parent.name, scope=parent.scope)
            if current.record != parent.record or current.path != parent.path:
                raise ArtifactError("parent artifact changed before stage registration")
        metadata_value = {} if metadata is None else dict(metadata)
        _canonical_json(metadata_value)
        records = {
            name: self._make_record(scope, stage, name, path)
            for name, path in sorted(outputs.items())
        }
        stage_value = {
            "config_sha256": self.config_sha256,
            "parents": [
                {
                    "artifact_id": item.record.artifact_id,
                    "name": item.name,
                    "scope": item.scope,
                    "sha256": item.record.sha256,
                    "stage": item.stage,
                }
                for item in parents
            ],
            "outputs": {name: asdict(record) for name, record in records.items()},
            "metadata": metadata_value,
        }
        state = self._load_state(scope)
        existing = state["stages"].get(stage)
        if existing is not None and existing != stage_value:
            raise ArtifactError(f"stage {stage} is already registered with different provenance")
        state["stages"][stage] = stage_value
        destination = self.state_path(scope)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return records

    def resolve(self, stage: str, name: str, *, scope: ArtifactScope) -> ResolvedArtifact:
        """Resolve a named parent output and revalidate its content; never scan the filesystem."""
        if not _SAFE_NAME.fullmatch(stage) or not _SAFE_NAME.fullmatch(name):
            raise ArtifactError("stage and artifact names must be safe nonempty names")
        state = self._load_state(scope)
        stage_value = state["stages"].get(stage)
        if not isinstance(stage_value, Mapping):
            raise ArtifactError(f"MISSING_PARENT_ARTIFACT: unregistered {scope} stage {stage}")
        outputs = stage_value.get("outputs")
        raw = outputs.get(name) if isinstance(outputs, Mapping) else None
        if not isinstance(raw, Mapping):
            raise ArtifactError(
                f"MISSING_PARENT_ARTIFACT: {scope} stage {stage} has no output {name}"
            )
        try:
            record = ArtifactRecord(**dict(raw))
        except TypeError as exc:
            raise ArtifactError("run state contains an invalid artifact record") from exc
        if record.kind not in {"file", "directory"} or len(record.sha256) != 64:
            raise ArtifactError("run state contains an invalid artifact record")
        candidate = self.root(scope) / record.path
        resolved, relative = self._relative_owned_path(scope, candidate)
        if relative != record.path:
            raise ArtifactError("run state artifact path is not canonical")
        digest, size, file_count, kind = hash_artifact(resolved)
        if (digest, size, file_count, kind) != (
            record.sha256,
            record.size_bytes,
            record.file_count,
            record.kind,
        ):
            raise ArtifactError(f"CACHE_HASH_MISMATCH: registered artifact changed: {resolved}")
        rebuilt = self._make_record(scope, stage, name, resolved)
        if rebuilt != record:
            raise ArtifactError("run state artifact identity is inconsistent")
        return ResolvedArtifact(stage=stage, name=name, scope=scope, path=resolved, record=record)
