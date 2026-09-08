from __future__ import annotations

import hashlib
import json
import os
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dive.artifacts import ArtifactError, ArtifactResolver, ResolvedArtifact
from dive.baseline import _atomic_json, _mapping
from dive.baseline_training import _repository_revision
from dive.cache import CacheError, make_cache_fingerprint
from dive.config import config_hash
from dive.data.manifest import load_manifest

from .audit import export_audit_template
from .proposals import load_proposals
from .slots import SCHEMA_VERSION, SchemaAudit


class AuditRunError(ValueError):
    """Human-audit export or decision provenance is incomplete or inconsistent."""


@dataclass(frozen=True)
class SemanticAuditDecision:
    schema_audit: SchemaAudit
    decision_path: Path
    decision_sha256: str
    annotation_path: Path
    annotation_sha256: str
    reviewed_pair_count: int


def _resolve_export_parents(resolver: ArtifactResolver) -> tuple[ResolvedArtifact, ...]:
    specifications = (
        ("validate_data", "train_manifest"),
        ("mine_propose", "proposals"),
        ("mine_propose", "report"),
    )
    try:
        return tuple(
            resolver.resolve(stage, name, scope="shared") for stage, name in specifications
        )
    except ArtifactError as exc:
        raise AuditRunError(
            "MISSING_PARENT_ARTIFACT: train proposals must be registered before audit export"
        ) from exc


def _parent_map(parents: Sequence[ResolvedArtifact]) -> dict[str, ResolvedArtifact]:
    return {f"{item.stage}.{item.name}": item for item in parents}


def _proposal_fingerprint(report: Mapping[str, Any]) -> str:
    raw = report.get("fingerprint")
    if not isinstance(raw, Mapping):
        raise AuditRunError("proposal report has no mining fingerprint")
    try:
        fingerprint = make_cache_fingerprint(str(raw["artifact"]), raw["components"])
    except (KeyError, TypeError, CacheError) as exc:
        raise AuditRunError("proposal report mining fingerprint is invalid") from exc
    if (
        fingerprint.artifact != "pair_scores"
        or fingerprint.to_dict() != dict(raw)
        or report.get("mining_fingerprint") != fingerprint.digest
    ):
        raise AuditRunError("CACHE_HASH_MISMATCH: proposal report fingerprint differs")
    return fingerprint.digest


def _load_proposal_report(parent: ResolvedArtifact, *, expected_config_hash: str) -> dict[str, Any]:
    try:
        report = json.loads(parent.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditRunError("proposal report is unreadable") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "dive_mine_propose.v1"
        or report.get("config_sha256") != expected_config_hash
        or report.get("test_content_used") is not False
        or report.get("semantic_audit_status") != "pending"
    ):
        raise AuditRunError("CACHE_HASH_MISMATCH: proposal report contract differs")
    _proposal_fingerprint(report)
    return report


def _pair_ids_hash(pair_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(pair_ids), sort_keys=False, separators=(",", ":")).encode()
    ).hexdigest()


def pre_audit_config_hash(config: Mapping[str, Any]) -> str:
    """Hash the locked upstream config before the human artifact path is populated."""
    value = copy.deepcopy(dict(config))
    mining = value.get("mining")
    if not isinstance(mining, dict):
        raise AuditRunError("config mining section is invalid")
    mining["schema_audit_artifact"] = None
    return config_hash(value)


def _registered_result(
    resolver: ArtifactResolver,
    destination: Path,
    *,
    configuration_sha: str,
    parent_hashes: Mapping[str, str],
    revision: str,
) -> tuple[dict[str, Any], bool] | None:
    try:
        report_parent = resolver.resolve("audit_export", "report", scope="shared")
    except ArtifactError as exc:
        if "MISSING_PARENT_ARTIFACT" not in str(exc):
            raise AuditRunError(str(exc)) from exc
        if not destination.exists():
            return None
        report_parent = None
    registered = report_parent is not None
    if registered:
        annotation = resolver.resolve("audit_export", "annotation_template", scope="shared")
        decision = resolver.resolve("audit_export", "decision_template", scope="shared")
        assert report_parent is not None
        report_path = report_parent.path
    else:
        annotation = None
        decision = None
        report_path = destination / "report.json"
        if not all(
            (destination / name).is_file()
            for name in ("annotation_template.jsonl", "decision_template.json", "report.json")
        ):
            raise AuditRunError(f"unregistered audit export is incomplete: {destination}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditRunError("audit export report is unreadable") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "dive_audit_export.v1"
        or report.get("config_sha256") != configuration_sha
        or report.get("parents") != dict(parent_hashes)
        or report.get("implementation_git_revision") != revision
        or report.get("test_content_used") is not False
    ):
        raise AuditRunError("CACHE_HASH_MISMATCH: audit export provenance differs")
    annotation_path = (
        annotation.path if annotation is not None else destination / "annotation_template.jsonl"
    )
    decision_path = (
        decision.path if decision is not None else destination / "decision_template.json"
    )
    if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != report.get(
        "annotation_sha256"
    ) or hashlib.sha256(decision_path.read_bytes()).hexdigest() != report.get(
        "decision_template_sha256"
    ):
        raise AuditRunError("audit export checksum differs from its report")
    return report, registered


def _register(
    resolver: ArtifactResolver,
    destination: Path,
    report: Mapping[str, Any],
    parents: Sequence[ResolvedArtifact],
) -> dict[str, Any]:
    registered = resolver.record_stage(
        "audit_export",
        {
            "annotation_template": destination / "annotation_template.jsonl",
            "decision_template": destination / "decision_template.json",
            "report": destination / "report.json",
        },
        scope="shared",
        parents=parents,
        metadata={
            "phase": "human_audit_export",
            "exported_size": int(report["exported_size"]),
            "ready_for_human_audit": bool(report["ready_for_human_audit"]),
        },
    )
    result = dict(report)
    result["artifacts"] = {
        name: {"artifact_id": item.artifact_id, "sha256": item.sha256}
        for name, item in registered.items()
    }
    return result


def export_proposal_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Export a deterministic blinded train proposal sample and an unfilled decision template."""
    run = _mapping(config, "run")
    data = _mapping(config, "data")
    mining = _mapping(config, "mining")
    if run.get("profile") != "correctness" or data.get("dataset") != "how2sign":
        raise AuditRunError("main audit export requires the How2Sign correctness profile")
    resolver = ArtifactResolver(config)
    parents = _resolve_export_parents(resolver)
    parent_by_name = _parent_map(parents)
    parent_hashes = {name: item.record.sha256 for name, item in sorted(parent_by_name.items())}
    configuration_sha = config_hash(config)
    revision = _repository_revision(Path(__file__).resolve().parents[5])
    destination = resolver.output_path("shared", "mining", "audit_export")
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

    records = load_manifest(
        parent_by_name["validate_data.train_manifest"].path, expected_split="train"
    )
    proposal_report = _load_proposal_report(
        parent_by_name["mine_propose.report"], expected_config_hash=configuration_sha
    )
    mining_fingerprint = _proposal_fingerprint(proposal_report)
    proposals = load_proposals(
        parent_by_name["mine_propose.proposals"].path,
        expected_fingerprint=mining_fingerprint,
        expected_sample_ids=[item.sample_id for item in records],
    )
    video_root = Path(str(data["video_root"]))
    context: dict[str, dict[str, str]] = {}
    for item in records:
        if item.video_path is None or not (video_root / item.video_path).is_file():
            raise AuditRunError(f"audit video is unavailable for sample {item.sample_id}")
        context[item.sample_id] = {
            "video_path": str((video_root / item.video_path).resolve()),
            "sign_language": item.sign_language,
            "text_language": item.text_language_model,
        }
    pending = destination.with_name(destination.name + ".pending")
    if pending.exists():
        raise AuditRunError(f"pending audit export already exists: {pending}")
    pending.mkdir(parents=True, exist_ok=False)
    export = export_audit_template(
        proposals,
        {item.sample_id: item.text_model for item in records},
        pending / "annotation_template.jsonl",
        sample_size=int(mining["shortlist_audit_queries"]),
        seed=int(mining["shortlist_audit_seed"]),
        sample_context=context,
    )
    decision_template = {
        "schema_version": "semantic_schema_audit_decision.v1",
        "schema_id": str(mining["semantic_rule_set"]),
        "audit_id": None,
        "audit_status": None,
        "proposal_mining_fingerprint": mining_fingerprint,
        "annotation_path": None,
        "annotation_sha256": None,
        "reviewed_pair_ids_sha256": _pair_ids_hash(export["pair_ids"]),
        "reviewed_pair_count": int(export["exported_size"]),
        "rating_scale": "integer_1_to_5_higher_means_caption_matches_sign",
        "decision_author": None,
        "decision_date_utc": None,
        "rationale": None,
        "scope": "strict_numeric_v1_how2sign_train_proposals",
    }
    _atomic_json(pending / "decision_template.json", decision_template)
    report = {
        "schema_version": "dive_audit_export.v1",
        "ready": True,
        "ready_for_human_audit": bool(export["exported_size"]),
        "config_sha256": configuration_sha,
        "implementation_git_revision": revision,
        "test_content_used": False,
        "parents": parent_hashes,
        "proposal_mining_fingerprint": mining_fingerprint,
        "proposal_count": len(proposals),
        "requested_size": int(export["requested_size"]),
        "exported_size": int(export["exported_size"]),
        "audit_seed": int(export["seed"]),
        "pair_ids": list(export["pair_ids"]),
        "reviewed_pair_ids_sha256": _pair_ids_hash(export["pair_ids"]),
        "annotation_sha256": str(export["artifact_sha256"]),
        "decision_template_sha256": hashlib.sha256(
            (pending / "decision_template.json").read_bytes()
        ).hexdigest(),
        "human_decision_required": True,
        "student_outputs_included": False,
    }
    _atomic_json(pending / "report.json", report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending, destination)
    return _register(resolver, destination, report, parents)


def _load_json_rows(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditRunError(f"{context} is unreadable") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditRunError(f"invalid {context} JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise AuditRunError(f"{context} row {line_number} must be an object")
        rows.append(value)
    return rows


def load_semantic_audit_decision(
    decision_path: str | Path,
    *,
    export_report_path: str | Path,
    annotation_template_path: str | Path,
    expected_mining_fingerprint: str,
) -> SemanticAuditDecision:
    """Validate a human decision and its filled annotation against the immutable export."""
    decision_source = Path(decision_path)
    export_source = Path(export_report_path)
    template_source = Path(annotation_template_path)
    if decision_source.is_symlink() or not decision_source.is_file():
        raise AuditRunError(f"semantic audit decision does not exist: {decision_source}")
    try:
        decision = json.loads(decision_source.read_text(encoding="utf-8"))
        export = json.loads(export_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditRunError("semantic audit decision/export report is unreadable") from exc
    expected_fields = {
        "schema_version",
        "schema_id",
        "audit_id",
        "audit_status",
        "proposal_mining_fingerprint",
        "annotation_path",
        "annotation_sha256",
        "reviewed_pair_ids_sha256",
        "reviewed_pair_count",
        "rating_scale",
        "decision_author",
        "decision_date_utc",
        "rationale",
        "scope",
    }
    if not isinstance(decision, dict) or set(decision) != expected_fields:
        raise AuditRunError("semantic audit decision fields are invalid")
    if not isinstance(export, Mapping) or export.get("schema_version") != "dive_audit_export.v1":
        raise AuditRunError("semantic audit export report schema is invalid")
    if (
        decision.get("schema_version") != "semantic_schema_audit_decision.v1"
        or decision.get("schema_id") != SCHEMA_VERSION
        or decision.get("audit_status") not in {"accepted", "rejected"}
        or decision.get("proposal_mining_fingerprint") != expected_mining_fingerprint
        or export.get("proposal_mining_fingerprint") != expected_mining_fingerprint
        or isinstance(decision.get("reviewed_pair_count"), bool)
        or not isinstance(decision.get("reviewed_pair_count"), int)
        or decision.get("reviewed_pair_count", 0) <= 0
        or decision.get("reviewed_pair_count") != export.get("exported_size")
        or decision.get("reviewed_pair_ids_sha256") != export.get("reviewed_pair_ids_sha256")
        or decision.get("rating_scale") != "integer_1_to_5_higher_means_caption_matches_sign"
        or decision.get("scope") != "strict_numeric_v1_how2sign_train_proposals"
    ):
        raise AuditRunError("semantic audit decision provenance/status differs from the export")
    for field in ("audit_id", "decision_author", "decision_date_utc", "rationale"):
        if not isinstance(decision.get(field), str) or not decision[field].strip():
            raise AuditRunError(f"semantic audit decision requires nonempty {field}")
    try:
        timestamp = datetime.fromisoformat(decision["decision_date_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditRunError("semantic audit decision date is not ISO-8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise AuditRunError("semantic audit decision date must be explicitly UTC")
    annotation_value = decision.get("annotation_path")
    if not isinstance(annotation_value, str) or not annotation_value:
        raise AuditRunError("semantic audit decision requires annotation_path")
    annotation_source = Path(annotation_value)
    if not annotation_source.is_absolute():
        annotation_source = decision_source.parent / annotation_source
    if annotation_source.is_symlink() or not annotation_source.is_file():
        raise AuditRunError(f"completed semantic annotation does not exist: {annotation_source}")
    annotation_path = annotation_source.resolve()
    if not annotation_path.is_file():
        raise AuditRunError(f"completed semantic annotation does not exist: {annotation_path}")
    annotation_sha = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    if decision.get("annotation_sha256") != annotation_sha:
        raise AuditRunError("completed semantic annotation checksum differs from decision")
    template_rows = _load_json_rows(template_source, "annotation template")
    completed_rows = _load_json_rows(annotation_path, "completed annotation")
    if (
        hashlib.sha256(template_source.read_bytes()).hexdigest() != export.get("annotation_sha256")
        or len(template_rows) != export.get("exported_size")
        or len(completed_rows) != len(template_rows)
    ):
        raise AuditRunError("completed semantic annotation count differs from export")
    mutable = {
        "category",
        "positive_i_rating",
        "positive_j_rating",
        "cross_i_j_negative_rating",
        "cross_j_i_negative_rating",
        "support_i_correct",
        "support_j_correct",
        "uncertain",
        "rater_id",
        "notes",
    }
    expected_pair_ids = tuple(map(str, export.get("pair_ids", ())))
    if tuple(row.get("pair_id") for row in completed_rows) != expected_pair_ids:
        raise AuditRunError("completed semantic annotation pair order differs from export")
    if _pair_ids_hash(expected_pair_ids) != decision["reviewed_pair_ids_sha256"]:
        raise AuditRunError("completed semantic annotation pair IDs differ from decision")
    for index, (template, completed) in enumerate(zip(template_rows, completed_rows, strict=True)):
        if set(template) != set(completed) or any(
            template[key] != completed[key] for key in set(template) - mutable
        ):
            raise AuditRunError(f"completed annotation changed immutable row {index}")
        for field in (
            "positive_i_rating",
            "positive_j_rating",
            "cross_i_j_negative_rating",
            "cross_j_i_negative_rating",
        ):
            value = completed.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise AuditRunError(f"completed annotation row {index} has invalid {field}")
        if not isinstance(completed.get("uncertain"), bool):
            raise AuditRunError(f"completed annotation row {index} requires uncertain bool")
        if not isinstance(completed.get("rater_id"), str) or not completed["rater_id"].strip():
            raise AuditRunError(f"completed annotation row {index} requires rater_id")
        if completed.get("category") is not None and not isinstance(completed["category"], str):
            raise AuditRunError(f"completed annotation row {index} category is invalid")
        for field in ("support_i_correct", "support_j_correct"):
            if completed.get(field) is not None and not isinstance(completed[field], bool):
                raise AuditRunError(f"completed annotation row {index} has invalid {field}")
        if completed.get("notes") is not None and not isinstance(completed["notes"], str):
            raise AuditRunError(f"completed annotation row {index} notes are invalid")
    decision_sha = hashlib.sha256(decision_source.read_bytes()).hexdigest()
    schema_audit = SchemaAudit(
        schema_id=SCHEMA_VERSION,
        audit_status=str(decision["audit_status"]),
        artifact_hash=decision_sha,
        audit_id=str(decision["audit_id"]),
    )
    schema_audit.validate()
    return SemanticAuditDecision(
        schema_audit=schema_audit,
        decision_path=decision_source.resolve(),
        decision_sha256=decision_sha,
        annotation_path=annotation_path,
        annotation_sha256=annotation_sha,
        reviewed_pair_count=len(completed_rows),
    )
