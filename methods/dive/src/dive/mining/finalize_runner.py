from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from dive.artifacts import ArtifactError, ArtifactResolver, ResolvedArtifact
from dive.baseline import _atomic_json, _mapping, _required_path
from dive.baseline_training import _repository_revision
from dive.cache import CacheError, make_cache_fingerprint
from dive.config import config_hash
from dive.data.manifest import load_manifest
from dive.data.text_units import load_text_unit_lineage

from .audit_runner import (
    AuditRunError,
    _load_proposal_report,
    _proposal_fingerprint,
    load_semantic_audit_decision,
    pre_audit_config_hash,
)
from .bank import BankError, finalize_numeric_bank, load_bank, write_bank
from .neighbors import NeighborError
from .proposals import load_proposals, proposal_from_dict, proposal_to_dict


class FinalizeRunError(ValueError):
    """Semantic bank finalization lacks an accepted, provenance-bound human audit."""


_OUTPUTS = ("pre_support_bank", "audit_snapshot", "semantic_rejections", "report")


def _resolve_parents(resolver: ArtifactResolver) -> tuple[ResolvedArtifact, ...]:
    specifications = (
        ("validate_data", "audit"),
        ("validate_data", "train_manifest"),
        ("validate_data", "text_unit_maps_dir"),
        ("mine_propose", "proposals"),
        ("mine_propose", "report"),
        ("audit_export", "annotation_template"),
        ("audit_export", "report"),
        ("baseline_validate", "report"),
        ("evidence_warmup", "reference"),
        ("cache_frozen_train", "reference_local"),
        ("cache_frozen_train", "report"),
    )
    try:
        return tuple(
            resolver.resolve(stage, name, scope="shared") for stage, name in specifications
        )
    except ArtifactError as exc:
        raise FinalizeRunError(
            "MISSING_PARENT_ARTIFACT: proposal, audit export, baseline/reference, and frozen "
            "support inputs must be registered before bank finalization"
        ) from exc


def _parent_map(parents: Sequence[ResolvedArtifact]) -> dict[str, ResolvedArtifact]:
    return {f"{item.stage}.{item.name}": item for item in parents}


def _load_audit_export_report(
    report_parent: ResolvedArtifact,
    annotation_parent: ResolvedArtifact,
    *,
    expected_config_hash: str,
    expected_mining_fingerprint: str,
) -> dict[str, Any]:
    try:
        report = json.loads(report_parent.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizeRunError("audit export report is unreadable") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "dive_audit_export.v1"
        or report.get("config_sha256") != expected_config_hash
        or report.get("proposal_mining_fingerprint") != expected_mining_fingerprint
        or report.get("test_content_used") is not False
        or report.get("student_outputs_included") is not False
        or report.get("annotation_sha256") != annotation_parent.record.sha256
    ):
        raise FinalizeRunError("CACHE_HASH_MISMATCH: audit export report differs")
    return report


def _fingerprint_from_report(report: Mapping[str, Any]) -> str:
    raw = report.get("fingerprint")
    if not isinstance(raw, Mapping):
        raise FinalizeRunError("finalized bank report has no fingerprint")
    try:
        fingerprint = make_cache_fingerprint(str(raw["artifact"]), raw["components"])
    except (KeyError, TypeError, CacheError) as exc:
        raise FinalizeRunError("finalized bank fingerprint is invalid") from exc
    if (
        fingerprint.artifact != "contrast_bank"
        or fingerprint.to_dict() != dict(raw)
        or report.get("bank_fingerprint") != fingerprint.digest
    ):
        raise FinalizeRunError("CACHE_HASH_MISMATCH: finalized bank fingerprint differs")
    return fingerprint.digest


def _registered_result(
    resolver: ArtifactResolver,
    destination: Path,
    *,
    configuration_sha: str,
    parent_hashes: Mapping[str, str],
    revision: str,
) -> tuple[dict[str, Any], bool] | None:
    try:
        report_parent = resolver.resolve("mine_finalize", "report", scope="shared")
    except ArtifactError as exc:
        if "MISSING_PARENT_ARTIFACT" not in str(exc):
            raise FinalizeRunError(str(exc)) from exc
        if not destination.exists():
            return None
        report_parent = None
    registered = report_parent is not None
    if registered:
        for name in _OUTPUTS:
            resolver.resolve("mine_finalize", name, scope="shared")
        assert report_parent is not None
        report_path = report_parent.path
    else:
        report_path = destination / "report.json"
        if any(
            not (destination / relative).exists()
            for relative in (
                "pre_support_bank",
                "audit_snapshot",
                "semantic_rejections.jsonl",
                "report.json",
            )
        ):
            raise FinalizeRunError(f"unregistered finalized bank is incomplete: {destination}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizeRunError("finalized bank report is unreadable") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "dive_mine_finalize.v1"
        or report.get("ready") is not True
        or report.get("config_sha256") != configuration_sha
        or report.get("implementation_git_revision") != revision
        or report.get("parents") != dict(parent_hashes)
        or report.get("test_content_used") is not False
    ):
        raise FinalizeRunError("CACHE_HASH_MISMATCH: finalized bank provenance differs")
    fingerprint = _fingerprint_from_report(report)
    if not registered:
        records = load_bank(destination / "pre_support_bank", fingerprint)
        decision = destination / "audit_snapshot" / "decision.json"
        annotation = destination / "audit_snapshot" / "completed_annotation.jsonl"
        rejection_path = destination / "semantic_rejections.jsonl"
        if not decision.is_file() or not annotation.is_file() or not rejection_path.is_file():
            raise FinalizeRunError("unregistered finalized bank snapshot is incomplete")
        try:
            rejection_rows = [
                json.loads(line)
                for line in rejection_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rejection_proposals = [
                proposal_from_dict(row["proposal"])
                for row in rejection_rows
                if isinstance(row, Mapping)
                and set(row) == {"schema_version", "reason", "proposal"}
                and row.get("schema_version") == "finalize_semantic_rejection.v1"
                and isinstance(row.get("reason"), str)
                and row["reason"]
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, NeighborError) as exc:
            raise FinalizeRunError("unregistered semantic rejection artifact is invalid") from exc
        if len(rejection_rows) != len(rejection_proposals):
            raise FinalizeRunError("unregistered semantic rejection artifact is invalid")
        rejection_counts = dict(sorted(Counter(row["reason"] for row in rejection_rows).items()))
        bank_pair_ids = {str(record.get("pair_id")) for record in records}
        rejected_pair_ids = {item.pair_id for item in rejection_proposals}
        if (
            len(records) != report.get("bank_record_count")
            or len(rejection_rows) != report.get("semantic_rejection_count")
            or rejection_counts != report.get("semantic_rejection_reasons")
            or bank_pair_ids & rejected_pair_ids
            or len(bank_pair_ids | rejected_pair_ids) != report.get("proposal_count")
            or report.get("ready_for_support") is not bool(records)
            or report.get("audit_status") != "accepted"
            or report.get("support_status") != "pending"
            or hashlib.sha256(decision.read_bytes()).hexdigest()
            != report.get("audit_decision_sha256")
            or hashlib.sha256(annotation.read_bytes()).hexdigest()
            != report.get("audit_annotation_sha256")
        ):
            raise FinalizeRunError("unregistered finalized bank outputs differ from report")
    return report, registered


def _register(
    resolver: ArtifactResolver,
    destination: Path,
    report: Mapping[str, Any],
    parents: Sequence[ResolvedArtifact],
) -> dict[str, Any]:
    registered = resolver.record_stage(
        "mine_finalize",
        {
            "pre_support_bank": destination / "pre_support_bank",
            "audit_snapshot": destination / "audit_snapshot",
            "semantic_rejections": destination / "semantic_rejections.jsonl",
            "report": destination / "report.json",
        },
        scope="shared",
        parents=parents,
        metadata={
            "phase": "finalize",
            "bank_record_count": int(report["bank_record_count"]),
            "audit_status": str(report["audit_status"]),
        },
    )
    result = dict(report)
    result["artifacts"] = {
        name: {"artifact_id": item.artifact_id, "sha256": item.sha256}
        for name, item in registered.items()
    }
    return result


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def finalize_train_contrasts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the immutable pre-support bank only after explicit accepted human audit."""
    run = _mapping(config, "run")
    data = _mapping(config, "data")
    mining = _mapping(config, "mining")
    support = _mapping(config, "support")
    if run.get("profile") != "correctness" or data.get("dataset") != "how2sign":
        raise FinalizeRunError("main bank finalization requires the How2Sign correctness profile")
    resolver = ArtifactResolver(config)
    parents = _resolve_parents(resolver)
    parent_by_name = _parent_map(parents)
    parent_hashes = {name: item.record.sha256 for name, item in sorted(parent_by_name.items())}
    configuration_sha = config_hash(config)
    revision = _repository_revision(Path(__file__).resolve().parents[5])
    destination = resolver.output_path("shared", "mining", "finalized")
    existing = _registered_result(
        resolver,
        destination,
        configuration_sha=configuration_sha,
        parent_hashes=parent_hashes,
        revision=revision,
    )
    if existing is not None:
        report, registered = existing
        return report if registered else _register(resolver, destination, report, parents)

    upstream_configuration_sha = pre_audit_config_hash(config)
    proposal_report = _load_proposal_report(
        parent_by_name["mine_propose.report"], expected_config_hash=upstream_configuration_sha
    )
    proposal_fingerprint = _proposal_fingerprint(proposal_report)
    audit_export = _load_audit_export_report(
        parent_by_name["audit_export.report"],
        parent_by_name["audit_export.annotation_template"],
        expected_config_hash=upstream_configuration_sha,
        expected_mining_fingerprint=proposal_fingerprint,
    )
    decision_path = _required_path(
        mining.get("schema_audit_artifact"), "mining.schema_audit_artifact"
    )
    try:
        decision = load_semantic_audit_decision(
            decision_path,
            export_report_path=parent_by_name["audit_export.report"].path,
            annotation_template_path=parent_by_name["audit_export.annotation_template"].path,
            expected_mining_fingerprint=proposal_fingerprint,
        )
    except AuditRunError as exc:
        raise FinalizeRunError(str(exc)) from exc
    if decision.schema_audit.audit_status != "accepted":
        raise FinalizeRunError("semantic schema audit was explicitly rejected; bank not created")
    records = load_manifest(
        parent_by_name["validate_data.train_manifest"].path, expected_split="train"
    )
    proposals = load_proposals(
        parent_by_name["mine_propose.proposals"].path,
        expected_fingerprint=proposal_fingerprint,
        expected_sample_ids=[item.sample_id for item in records],
    )
    expected_texts: dict[str, str] = {}
    for record in records:
        previous = expected_texts.setdefault(record.text_id, record.text_model)
        if previous != record.text_model:
            raise FinalizeRunError(f"train text ID has inconsistent captions: {record.text_id}")
    lineages = load_text_unit_lineage(
        parent_by_name["validate_data.text_unit_maps_dir"].path / "train.jsonl",
        expected_texts=expected_texts,
    )
    fingerprint = make_cache_fingerprint(
        "contrast_bank",
        {
            "train_manifest": parent_by_name["validate_data.train_manifest"].record.sha256,
            "baseline": {
                "validation": parent_by_name["baseline_validate.report"].record.sha256,
                "native_cache": parent_by_name["cache_frozen_train.report"].record.sha256,
            },
            "reference": parent_by_name["evidence_warmup.reference"].record.sha256,
            "schema_audit": {
                "decision": decision.decision_sha256,
                "annotation": decision.annotation_sha256,
                "reviewed_pair_count": decision.reviewed_pair_count,
            },
            "units": parent_by_name["validate_data.text_unit_maps_dir"].record.sha256,
            "mining_config": {
                **{**dict(mining), "schema_audit_artifact": None},
                "proposal_fingerprint": proposal_fingerprint,
            },
            "support_config": dict(support),
            "timestamps": parent_by_name["cache_frozen_train.reference_local"].record.sha256,
            "implementation_git_revision": revision,
        },
    )
    try:
        build = finalize_numeric_bank(
            proposals,
            text_by_sample={item.sample_id: item.text_model for item in records},
            split_by_sample={item.sample_id: item.split for item in records},
            unit_mapping_hashes={
                item.sample_id: lineages[item.text_id].unit_mapping_sha256 for item in records
            },
            audit=decision.schema_audit,
            mining_fingerprint=fingerprint.digest,
        )
    except (BankError, KeyError) as exc:
        raise FinalizeRunError(f"semantic bank construction failed: {exc}") from exc
    pending = destination.with_name(destination.name + ".pending")
    build_state = {
        "schema_version": "dive_mine_finalize_build.v1",
        "config_sha256": configuration_sha,
        "implementation_git_revision": revision,
        "parents": parent_hashes,
        "bank_fingerprint": fingerprint.digest,
        "audit_decision_sha256": decision.decision_sha256,
        "audit_annotation_sha256": decision.annotation_sha256,
    }
    if pending.exists():
        try:
            stored_state = json.loads((pending / "build_state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FinalizeRunError("pending finalized-bank state is unreadable") from exc
        if stored_state != build_state:
            raise FinalizeRunError("CACHE_HASH_MISMATCH: pending finalized-bank state differs")
    else:
        pending.mkdir(parents=True, exist_ok=False)
        _atomic_json(pending / "build_state.json", build_state)
    write_bank(build.records, pending / "pre_support_bank", fingerprint.digest)
    _atomic_copy(decision.decision_path, pending / "audit_snapshot" / "decision.json")
    _atomic_copy(
        decision.annotation_path, pending / "audit_snapshot" / "completed_annotation.jsonl"
    )
    rejected_rows = []
    proposal_by_id = {item.pair_id: item for item in proposals}
    for rejected in build.rejected:
        proposal = proposal_by_id.get(rejected["pair_id"])
        if proposal is None:
            raise FinalizeRunError("semantic rejection references an unknown proposal")
        rejected_rows.append(
            {
                "schema_version": "finalize_semantic_rejection.v1",
                "reason": rejected["reason"],
                "proposal": proposal_to_dict(proposal),
            }
        )
    rejection_path = pending / "semantic_rejections.jsonl"
    rejection_tmp = rejection_path.with_suffix(rejection_path.suffix + ".tmp")
    rejection_tmp.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rejected_rows
        ),
        encoding="utf-8",
    )
    os.replace(rejection_tmp, rejection_path)
    report = {
        "schema_version": "dive_mine_finalize.v1",
        "ready": True,
        "ready_for_support": bool(build.records),
        "config_sha256": configuration_sha,
        "implementation_git_revision": revision,
        "test_content_used": False,
        "parents": parent_hashes,
        "proposal_mining_fingerprint": proposal_fingerprint,
        "bank_fingerprint": fingerprint.digest,
        "fingerprint": fingerprint.to_dict(),
        "proposal_count": len(proposals),
        "bank_record_count": len(build.records),
        "semantic_rejection_count": len(build.rejected),
        "semantic_rejection_reasons": dict(
            sorted(Counter(item["reason"] for item in build.rejected).items())
        ),
        "audit_status": decision.schema_audit.audit_status,
        "audit_id": decision.schema_audit.audit_id,
        "audit_decision_sha256": decision.decision_sha256,
        "audit_annotation_sha256": decision.annotation_sha256,
        "audit_reviewed_pair_count": decision.reviewed_pair_count,
        "audit_exported_pair_count": int(audit_export["exported_size"]),
        "schema_id": decision.schema_audit.schema_id,
        "support_status": "pending",
    }
    _atomic_json(pending / "report.json", report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending, destination)
    return _register(resolver, destination, report, parents)
