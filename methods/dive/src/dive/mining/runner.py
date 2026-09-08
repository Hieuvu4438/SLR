from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from dive.artifacts import ArtifactError, ArtifactResolver, ResolvedArtifact
from dive.baseline import _atomic_json, _mapping
from dive.baseline_training import _repository_revision
from dive.cache import CacheError, CacheFingerprint, load_tensor_cache, make_cache_fingerprint
from dive.config import config_hash
from dive.data.manifest import SampleRecord, load_manifest
from dive.data.relations import load_excluded_negatives
from dive.data.relevance import load_relevance
from dive.data.text_units import TextUnitError, load_text_unit_lineage, require_target

from .neighbors import NeighborError, NeighborProposal
from .pair_scorer import PersistentSedsPairScorer, validate_persistent_pair_score_cache
from .proposals import (
    load_proposals,
    proposal_from_dict,
    proposal_to_dict,
    write_proposals,
)
from .scalable import (
    PooledShortlists,
    audit_sparse_shortlist_coverage,
    pooled_shortlists_blockwise,
    rerank_sparse_shortlists,
)
from .slots import SCHEMA_VERSION, SchemaAudit, validate_strict_numeric_pair


class MiningRunError(ValueError):
    """Executable mining violated cache, split, or provenance contracts."""


_OUTPUT_NAMES = (
    "proposals",
    "shortlists",
    "shortlists_meta",
    "pair_scores",
    "shortlist_coverage",
    "semantic_rejections",
    "report",
)


def _resolve_propose_parents(
    resolver: ArtifactResolver,
) -> tuple[ResolvedArtifact, ...]:
    specifications = (
        ("validate_data", "audit"),
        ("validate_data", "train_manifest"),
        ("validate_data", "train_relations"),
        ("validate_data", "train_relevance"),
        ("validate_data", "text_unit_maps_dir"),
        ("cache_frozen_train", "native_video"),
        ("cache_frozen_train", "native_text"),
        ("cache_frozen_train", "report"),
    )
    try:
        return tuple(
            resolver.resolve(stage, name, scope="shared") for stage, name in specifications
        )
    except ArtifactError as exc:
        raise MiningRunError(
            "MISSING_PARENT_ARTIFACT: validated train relations and frozen native caches must be "
            "registered before proposal mining"
        ) from exc


def _parent_map(parents: Sequence[ResolvedArtifact]) -> dict[str, ResolvedArtifact]:
    return {f"{item.stage}.{item.name}": item for item in parents}


def build_forbidden_candidates(
    records: Sequence[SampleRecord],
    relevance: Mapping[str, Collection[str]],
    excluded_pairs: Sequence[tuple[str, str]],
) -> tuple[frozenset[int], ...]:
    """Map video-to-text relation IDs onto sample-addressed paired candidate rows."""
    text_rows: dict[str, set[int]] = {}
    for index, record in enumerate(records):
        text_rows.setdefault(record.text_id, set()).add(index)
    excluded_by_video: dict[str, set[str]] = {}
    for video_id, text_id in excluded_pairs:
        excluded_by_video.setdefault(video_id, set()).add(text_id)
    output: list[frozenset[int]] = []
    for index, record in enumerate(records):
        if record.video_id not in relevance:
            raise MiningRunError(f"train relevance is missing video ID {record.video_id}")
        forbidden_texts = set(relevance[record.video_id]) | excluded_by_video.get(
            record.video_id, set()
        )
        rows: set[int] = {index}
        for text_id in forbidden_texts:
            if text_id not in text_rows:
                raise MiningRunError(f"train relation references unknown text ID {text_id}")
            rows.update(text_rows[text_id])
        output.append(frozenset(rows))
    return tuple(output)


def _load_cache_fingerprint(raw: Any, name: str) -> CacheFingerprint:
    if not isinstance(raw, Mapping):
        raise MiningRunError(f"frozen cache report lacks {name} fingerprint")
    try:
        fingerprint = make_cache_fingerprint(str(raw["artifact"]), raw["components"])
    except (KeyError, TypeError, CacheError) as exc:
        raise MiningRunError(f"frozen cache {name} fingerprint is invalid") from exc
    if fingerprint.to_dict() != dict(raw):
        raise MiningRunError(f"CACHE_HASH_MISMATCH: frozen cache {name} fingerprint differs")
    return fingerprint


def _load_native_features(
    video_parent: ResolvedArtifact,
    text_parent: ResolvedArtifact,
    report_parent: ResolvedArtifact,
    records: Sequence[SampleRecord],
    *,
    expected_config_hash: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Mapping[str, Any]]:
    try:
        report = json.loads(report_parent.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MiningRunError("frozen cache report is unreadable") from exc
    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != "dive_frozen_train_cache.v1"
        or report.get("config_sha256") != expected_config_hash
        or report.get("kind") != "frozen_train"
        or report.get("test_content_used") is not False
    ):
        raise MiningRunError("CACHE_HASH_MISMATCH: frozen train cache report differs")
    fingerprints = report.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise MiningRunError("frozen cache report lacks fingerprints")
    video_fingerprint = _load_cache_fingerprint(fingerprints.get("native_video"), "native_video")
    text_fingerprint = _load_cache_fingerprint(fingerprints.get("native_text"), "native_text")
    sample_ids = tuple(item.sample_id for item in records)
    video_shards = load_tensor_cache(
        video_parent.path,
        expected_namespace="train_native_video",
        expected_fingerprint=video_fingerprint,
        expected_ordered_ids=sample_ids,
    )
    text_shards = load_tensor_cache(
        text_parent.path,
        expected_namespace="train_native_text",
        expected_fingerprint=text_fingerprint,
        expected_ordered_ids=sample_ids,
    )
    if any(
        set(shard.tensors) != {"pooled", "fusion_hidden"}
        or set(shard.masks) != {"video"}
        or shard.metadata.get("split") != "train"
        or shard.metadata.get("contains_cls") is not True
        for shard in video_shards
    ):
        raise MiningRunError("native video cache fields/metadata violate the mining contract")
    if any(
        set(shard.tensors) != {"pooled", "tokens"}
        or set(shard.masks) != {"text"}
        or shard.metadata.get("split") != "train"
        for shard in text_shards
    ):
        raise MiningRunError("native text cache fields/metadata violate the mining contract")
    stored_text_ids: list[str] = []
    for shard in text_shards:
        values = shard.metadata.get("text_ids")
        if not isinstance(values, list) or len(values) != len(shard.ordered_ids):
            raise MiningRunError("native text cache lacks sample-aligned text IDs")
        stored_text_ids.extend(map(str, values))
    if tuple(stored_text_ids) != tuple(item.text_id for item in records):
        raise MiningRunError("CACHE_ORDER_MISMATCH: native text IDs differ from the train manifest")
    try:
        pooled_video = torch.cat([shard.tensors["pooled"] for shard in video_shards])
        fusion_hidden = torch.cat([shard.tensors["fusion_hidden"] for shard in video_shards])
        video_mask = torch.cat([shard.masks["video"] for shard in video_shards])
        pooled_text = torch.cat([shard.tensors["pooled"] for shard in text_shards])
        text_hidden = torch.cat([shard.tensors["tokens"] for shard in text_shards])
        text_mask = torch.cat([shard.masks["text"] for shard in text_shards])
    except RuntimeError as exc:
        raise MiningRunError("native cache shards cannot be concatenated consistently") from exc
    if (
        fusion_hidden.shape[0] != len(records)
        or text_hidden.shape[0] != len(records)
        or fusion_hidden.shape[-1] != text_hidden.shape[-1]
    ):
        raise MiningRunError("native cache dimensions do not align with the train manifest")
    return (
        pooled_video,
        pooled_text,
        fusion_hidden,
        text_hidden,
        video_mask,
        text_mask,
        report,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_shortlists(
    path: Path,
    metadata_path: Path,
    shortlists: PooledShortlists,
    sample_ids: Sequence[str],
    fingerprint: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            video_to_text=shortlists.video_to_text,
            text_to_video=shortlists.text_to_video,
            sample_ids=np.asarray(sample_ids),
        )
    os.replace(temporary, path)
    _atomic_json(
        metadata_path,
        {
            "schema_version": "pooled_shortlists.v1",
            "fingerprint": fingerprint,
            "sample_count": len(sample_ids),
            "shortlist_per_direction": int(shortlists.video_to_text.shape[1]),
            "shortlists_sha256": _sha256(path),
        },
    )


def _load_shortlists(
    path: Path,
    metadata_path: Path,
    sample_ids: Sequence[str],
    fingerprint: str,
    limit: int,
) -> PooledShortlists:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MiningRunError("pooled shortlist metadata is unreadable") from exc
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != "pooled_shortlists.v1"
        or metadata.get("fingerprint") != fingerprint
        or metadata.get("sample_count") != len(sample_ids)
        or metadata.get("shortlist_per_direction") != limit
        or metadata.get("shortlists_sha256") != _sha256(path)
    ):
        raise MiningRunError("CACHE_HASH_MISMATCH: pooled shortlist provenance differs")
    try:
        with np.load(path, allow_pickle=False) as payload:
            stored_ids = tuple(map(str, payload["sample_ids"].tolist()))
            video_to_text = np.asarray(payload["video_to_text"], dtype=np.int64)
            text_to_video = np.asarray(payload["text_to_video"], dtype=np.int64)
    except (OSError, ValueError, KeyError) as exc:
        raise MiningRunError("pooled shortlist data is unreadable") from exc
    expected_shape = (len(sample_ids), limit)
    if (
        stored_ids != tuple(sample_ids)
        or video_to_text.shape != expected_shape
        or text_to_video.shape != expected_shape
    ):
        raise MiningRunError("CACHE_ORDER_MISMATCH: pooled shortlists differ from train IDs")
    return PooledShortlists(video_to_text, text_to_video)


def _semantic_candidates(
    proposals: Sequence[NeighborProposal],
    records: Sequence[SampleRecord],
    lineages: Mapping[str, Any],
) -> tuple[tuple[NeighborProposal, ...], tuple[dict[str, Any], ...]]:
    by_sample = {item.sample_id: item for item in records}
    audit = SchemaAudit(SCHEMA_VERSION, "pending", None, f"{SCHEMA_VERSION}_proposal")
    accepted: list[NeighborProposal] = []
    rejected: list[dict[str, Any]] = []

    def reject(proposal: NeighborProposal, reason: str) -> None:
        rejected.append(
            {
                "schema_version": "semantic_rejection.v1",
                "reason": reason,
                "proposal": proposal_to_dict(proposal),
            }
        )

    for proposal in proposals:
        left = by_sample[proposal.sample_i]
        right = by_sample[proposal.sample_j]
        result = validate_strict_numeric_pair(left.text_model, right.text_model, audit)
        if not result.eligible:
            reject(proposal, result.reason)
            continue
        assert result.unit_i is not None and result.unit_j is not None
        try:
            require_target(lineages[left.text_id].units, result.unit_i)
            require_target(lineages[right.text_id].units, result.unit_j)
        except (KeyError, TextUnitError):
            reject(proposal, "target_unit_incomplete_after_truncation")
            continue
        accepted.append(proposal)
    return tuple(accepted), tuple(rejected)


def _write_rejections(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _registered_result(
    resolver: ArtifactResolver,
    destination: Path,
    *,
    configuration_sha: str,
    parents: Mapping[str, str],
    revision: str,
    sample_ids: Sequence[str],
    shortlist_limit: int,
) -> tuple[dict[str, Any], bool] | None:
    try:
        report_parent = resolver.resolve("mine_propose", "report", scope="shared")
    except ArtifactError as exc:
        if "MISSING_PARENT_ARTIFACT" not in str(exc):
            raise MiningRunError(str(exc)) from exc
        if not destination.exists():
            return None
        report_parent = None
    registered = report_parent is not None
    if registered:
        for name in _OUTPUT_NAMES:
            resolver.resolve("mine_propose", name, scope="shared")
        report_path = report_parent.path
    else:
        report_path = destination / "report.json"
        if any(
            not (destination / relative).exists()
            for relative in (
                "proposals",
                "shortlists.npz",
                "shortlists_meta.json",
                "pair_scores",
                "shortlist_coverage.json",
                "semantic_rejections.jsonl",
                "report.json",
            )
        ):
            raise MiningRunError(f"unregistered proposal destination is incomplete: {destination}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MiningRunError("proposal mining report is unreadable") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "dive_mine_propose.v1"
        or report.get("ready") is not True
        or report.get("phase") != "propose"
        or report.get("test_content_used") is not False
        or report.get("config_sha256") != configuration_sha
        or report.get("parents") != dict(parents)
        or report.get("implementation_git_revision") != revision
    ):
        raise MiningRunError("CACHE_HASH_MISMATCH: proposal mining provenance differs")
    report_fingerprint = _load_cache_fingerprint(report.get("fingerprint"), "pair_scores")
    if (
        report_fingerprint.artifact != "pair_scores"
        or report.get("mining_fingerprint") != report_fingerprint.digest
    ):
        raise MiningRunError("CACHE_HASH_MISMATCH: proposal mining fingerprint differs")
    if not registered:
        mining_fingerprint = report_fingerprint.digest
        proposals = load_proposals(
            destination / "proposals",
            expected_fingerprint=mining_fingerprint,
            expected_sample_ids=sample_ids,
        )
        _load_shortlists(
            destination / "shortlists.npz",
            destination / "shortlists_meta.json",
            sample_ids,
            mining_fingerprint,
            shortlist_limit,
        )
        score_cache = validate_persistent_pair_score_cache(
            destination / "pair_scores", expected_fingerprint=mining_fingerprint
        )
        try:
            coverage = json.loads(
                (destination / "shortlist_coverage.json").read_text(encoding="utf-8")
            )
            rejection_rows = [
                json.loads(line)
                for line in (destination / "semantic_rejections.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise MiningRunError("unregistered proposal diagnostics are unreadable") from exc
        coverage_anchors = coverage.get("anchors") if isinstance(coverage, Mapping) else None
        if (
            not isinstance(coverage, Mapping)
            or not isinstance(coverage_anchors, Mapping)
            or coverage.get("schema_version") != "shortlist_coverage.v1"
            or coverage.get("mining_fingerprint") != mining_fingerprint
            or coverage.get("coverage") != report.get("shortlist_coverage")
            or coverage.get("hits") != report.get("shortlist_coverage_hits")
            or coverage.get("total") != report.get("shortlist_coverage_total")
            or set(coverage_anchors) != set(report.get("shortlist_audit_query_ids", ()))
            or len(proposals) != report.get("eligible_proposal_count")
            or score_cache["score_count"] <= 0
        ):
            raise MiningRunError("unregistered proposal diagnostics differ from the report")
        try:
            rejection_proposals = [
                proposal_from_dict(row["proposal"])
                for row in rejection_rows
                if isinstance(row, Mapping)
                and set(row) == {"schema_version", "reason", "proposal"}
                and row.get("schema_version") == "semantic_rejection.v1"
                and isinstance(row.get("reason"), str)
                and row["reason"]
            ]
        except (KeyError, TypeError, NeighborError) as exc:
            raise MiningRunError("unregistered semantic rejection rows are invalid") from exc
        allowed_sample_ids = set(sample_ids)
        if len(rejection_proposals) != len(rejection_rows) or any(
            item.sample_i not in allowed_sample_ids or item.sample_j not in allowed_sample_ids
            for item in rejection_proposals
        ):
            raise MiningRunError("unregistered semantic rejection rows are invalid")
        rejection_counts = dict(sorted(Counter(row["reason"] for row in rejection_rows).items()))
        if len(rejection_rows) != report.get(
            "semantic_rejection_count"
        ) or rejection_counts != report.get("semantic_rejection_reasons"):
            raise MiningRunError("unregistered semantic rejection counts differ from the report")
    return report, registered


def _register(
    resolver: ArtifactResolver,
    destination: Path,
    report: Mapping[str, Any],
    parents: Sequence[ResolvedArtifact],
) -> dict[str, Any]:
    registered = resolver.record_stage(
        "mine_propose",
        {
            "proposals": destination / "proposals",
            "shortlists": destination / "shortlists.npz",
            "shortlists_meta": destination / "shortlists_meta.json",
            "pair_scores": destination / "pair_scores",
            "shortlist_coverage": destination / "shortlist_coverage.json",
            "semantic_rejections": destination / "semantic_rejections.jsonl",
            "report": destination / "report.json",
        },
        scope="shared",
        parents=parents,
        metadata={
            "phase": "propose",
            "eligible_proposal_count": int(report["eligible_proposal_count"]),
            "coverage": float(report["shortlist_coverage"]),
        },
    )
    result = dict(report)
    result["artifacts"] = {
        name: {"artifact_id": item.artifact_id, "sha256": item.sha256}
        for name, item in registered.items()
    }
    return result


def propose_train_contrasts(
    config: Mapping[str, Any],
    *,
    device: str = "cuda:0",
    pooled_query_chunk_size: int = 512,
    pair_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Mine train-only strict-schema candidates and exact shortlist coverage."""
    if pooled_query_chunk_size <= 0 or pair_chunk_size <= 0:
        raise MiningRunError("mining chunk sizes must be positive")
    run = _mapping(config, "run")
    data = _mapping(config, "data")
    mining = _mapping(config, "mining")
    baseline = _mapping(config, "baseline")
    evidence = _mapping(config, "evidence")
    if run.get("profile") != "correctness" or data.get("dataset") != "how2sign":
        raise MiningRunError("main proposal mining requires the How2Sign correctness profile")
    if baseline.get("family") != "seds" or baseline.get("score_scale") != "prelogit":
        raise MiningRunError("proposal mining currently requires the SEDS prelogit baseline")
    if mining.get("semantic_rule_set") != SCHEMA_VERSION:
        raise MiningRunError(f"proposal mining requires semantic_rule_set={SCHEMA_VERSION}")
    resolver = ArtifactResolver(config)
    parents = _resolve_propose_parents(resolver)
    parent_by_name = _parent_map(parents)
    parent_hashes = {name: item.record.sha256 for name, item in sorted(parent_by_name.items())}
    configuration_sha = config_hash(config)
    revision = _repository_revision(Path(__file__).resolve().parents[5])
    destination = resolver.output_path("shared", "mining", "propose")
    manifest_parent = parent_by_name["validate_data.train_manifest"]
    relation_parent = parent_by_name["validate_data.train_relations"]
    relevance_parent = parent_by_name["validate_data.train_relevance"]
    unit_parent = parent_by_name["validate_data.text_unit_maps_dir"]
    video_parent = parent_by_name["cache_frozen_train.native_video"]
    text_parent = parent_by_name["cache_frozen_train.native_text"]
    cache_report_parent = parent_by_name["cache_frozen_train.report"]
    records = load_manifest(manifest_parent.path, expected_split="train")
    existing = _registered_result(
        resolver,
        destination,
        configuration_sha=configuration_sha,
        parents=parent_hashes,
        revision=revision,
        sample_ids=[item.sample_id for item in records],
        shortlist_limit=int(mining["shortlist_per_direction"]),
    )
    if existing is not None:
        report, registered = existing
        return report if registered else _register(resolver, destination, report, parents)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise MiningRunError("CUDA mining was requested but CUDA is unavailable")

    video_ids = tuple(item.video_id for item in records)
    text_ids = tuple(item.text_id for item in records)
    relevance = load_relevance(relevance_parent.path, video_ids=video_ids, text_ids=text_ids)
    exclusions = load_excluded_negatives(
        relation_parent.path,
        video_ids=video_ids,
        text_ids=text_ids,
        positives_by_video=relevance,
    )
    forbidden = build_forbidden_candidates(
        records, relevance, [(item.video_id, item.text_id) for item in exclusions]
    )
    expected_texts: dict[str, str] = {}
    for record in records:
        previous = expected_texts.setdefault(record.text_id, record.text_model)
        if previous != record.text_model:
            raise MiningRunError(f"train text ID has inconsistent captions: {record.text_id}")
    lineages = load_text_unit_lineage(
        unit_parent.path / "train.jsonl",
        expected_texts=expected_texts,
    )
    (
        pooled_video,
        pooled_text,
        fusion_hidden,
        text_hidden,
        video_mask,
        text_mask,
        cache_report,
    ) = _load_native_features(
        video_parent,
        text_parent,
        cache_report_parent,
        records,
        expected_config_hash=configuration_sha,
    )
    fingerprint = make_cache_fingerprint(
        "pair_scores",
        {
            "feature_hashes": {
                "native_video": video_parent.record.sha256,
                "native_text": text_parent.record.sha256,
                "train_manifest": manifest_parent.record.sha256,
                "train_relations": relation_parent.record.sha256,
                "train_relevance": relevance_parent.record.sha256,
            },
            "scorer_version": "seds_masked_aligned_prelogit.v1",
            "tau_alignment": float(evidence["tau_alignment"]),
            "native_mixing": float(baseline["dual_mix"]),
            "score_scale": "prelogit",
            "pooling": "mean_normalized_contextual_then_l2",
            "shortlist_per_direction": int(mining["shortlist_per_direction"]),
            "rerank_topk": int(mining["rerank_topk"]),
            "implementation_git_revision": revision,
        },
    )
    pending = destination.with_name(destination.name + ".pending")
    state = {
        "schema_version": "dive_mine_propose_build.v1",
        "config_sha256": configuration_sha,
        "implementation_git_revision": revision,
        "parents": parent_hashes,
        "mining_fingerprint": fingerprint.digest,
        "sample_count": len(records),
        "pooled_query_chunk_size": pooled_query_chunk_size,
        "pair_chunk_size": pair_chunk_size,
    }
    if pending.exists():
        try:
            stored_state = json.loads((pending / "build_state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MiningRunError("pending proposal build state is unreadable") from exc
        if stored_state != state:
            raise MiningRunError("CACHE_HASH_MISMATCH: pending proposal provenance differs")
    else:
        pending.mkdir(parents=True, exist_ok=False)
        _atomic_json(pending / "build_state.json", state)

    shortlist_path = pending / "shortlists.npz"
    shortlist_meta_path = pending / "shortlists_meta.json"
    if shortlist_path.exists() or shortlist_meta_path.exists():
        if not shortlist_path.exists() or not shortlist_meta_path.exists():
            raise MiningRunError("pending pooled shortlist artifact is incomplete")
        shortlists = _load_shortlists(
            shortlist_path,
            shortlist_meta_path,
            [item.sample_id for item in records],
            fingerprint.digest,
            int(mining["shortlist_per_direction"]),
        )
    else:
        shortlists = pooled_shortlists_blockwise(
            pooled_video,
            pooled_text,
            [item.sample_id for item in records],
            forbidden,
            limit=int(mining["shortlist_per_direction"]),
            query_chunk_size=pooled_query_chunk_size,
            device=target,
        )
        _write_shortlists(
            shortlist_path,
            shortlist_meta_path,
            shortlists,
            [item.sample_id for item in records],
            fingerprint.digest,
        )
    scorer = PersistentSedsPairScorer(
        fusion_hidden,
        text_hidden,
        video_mask,
        text_mask,
        dual_mix=float(baseline["dual_mix"]),
        temperature=float(evidence["tau_alignment"]),
        device=target,
        pair_chunk_size=pair_chunk_size,
        cache_dir=pending / "pair_scores",
        fingerprint=fingerprint.digest,
    )
    reranked = rerank_sparse_shortlists(
        shortlists,
        [item.sample_id for item in records],
        forbidden,
        scorer,
        rerank_topk=int(mining["rerank_topk"]),
    )
    audit_count = int(mining["shortlist_audit_queries"])
    if audit_count <= 0 or audit_count > len(records):
        raise MiningRunError("shortlist audit query count is invalid for the train manifest")
    randomizer = random.Random(int(mining["shortlist_audit_seed"]))
    audit_ids = tuple(
        sorted(randomizer.sample(sorted(item.sample_id for item in records), audit_count))
    )
    coverage = audit_sparse_shortlist_coverage(
        shortlists,
        [item.sample_id for item in records],
        forbidden,
        audit_ids,
        scorer,
        reranked.self_scores,
        hard_topk=int(mining["rerank_topk"]),
    )
    coverage.update(
        {
            "audit_seed": int(mining["shortlist_audit_seed"]),
            "shortlist_per_direction": int(mining["shortlist_per_direction"]),
            "hard_topk": int(mining["rerank_topk"]),
            "mining_fingerprint": fingerprint.digest,
        }
    )
    _atomic_json(pending / "shortlist_coverage.json", coverage)
    eligible, rejected = _semantic_candidates(reranked.proposals, records, lineages)
    write_proposals(eligible, pending / "proposals", mining_fingerprint=fingerprint.digest)
    _write_rejections(pending / "semantic_rejections.jsonl", rejected)
    rejection_counts = dict(sorted(Counter(item["reason"] for item in rejected).items()))
    report = {
        "schema_version": "dive_mine_propose.v1",
        "ready": True,
        "phase": "propose",
        "config_sha256": configuration_sha,
        "implementation_git_revision": revision,
        "test_content_used": False,
        "parents": parent_hashes,
        "mining_fingerprint": fingerprint.digest,
        "fingerprint": fingerprint.to_dict(),
        "sample_count": len(records),
        "known_positive_pair_count": sum(len(values) for values in relevance.values()),
        "excluded_negative_count": len(exclusions),
        "shortlisted_pair_count": reranked.shortlisted_pair_count,
        "reranked_proposal_count": len(reranked.proposals),
        "eligible_proposal_count": len(eligible),
        "semantic_rejection_count": len(rejected),
        "semantic_rejection_reasons": rejection_counts,
        "semantic_rule_set": SCHEMA_VERSION,
        "semantic_audit_status": "pending",
        "shortlist_audit_query_ids": list(audit_ids),
        "shortlist_coverage": float(coverage["coverage"]),
        "shortlist_coverage_hits": int(coverage["hits"]),
        "shortlist_coverage_total": int(coverage["total"]),
        "pair_score_dtype": "float32",
        "cache_implementation_git_revision": cache_report.get("implementation_git_revision"),
        "chunking": {
            "pooled_query_chunk_size": pooled_query_chunk_size,
            "pair_chunk_size": pair_chunk_size,
        },
    }
    _atomic_json(pending / "report.json", report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending, destination)
    return _register(resolver, destination, report, parents)
