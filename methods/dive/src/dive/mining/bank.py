from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .neighbors import NeighborProposal
from .slots import SchemaAudit, validate_strict_numeric_pair
from .support import PairSupport


class BankError(ValueError):
    """A contrast bank is stale, corrupt, or violates train-only provenance."""


@dataclass(frozen=True)
class BankRecord:
    schema_version: str
    pair_id: str
    sample_i: str
    sample_j: str
    text_hash_i: str
    text_hash_j: str
    unit_i: int
    unit_j: int
    unit_mapping_hash_i: str
    unit_mapping_hash_j: str
    category: str
    schema_id: str
    audit_hash: str
    g_sem: float
    s0_quartet: tuple[tuple[float, float], tuple[float, float]]
    margins0: tuple[float, float, float, float]
    hardness: float
    reference_hash: str | None
    grid_i: str | None
    grid_j: str | None
    views_i: tuple[str, ...]
    views_j: tuple[str, ...]
    q_i: tuple[float, ...] | None
    q_j: tuple[float, ...] | None
    h: float | None
    stability_i: float | None
    stability_j: float | None
    g: float
    retained_mass_i: float | None
    retained_mass_j: float | None
    rf_union_ratio_i: float | None
    rf_union_ratio_j: float | None
    support_status: str
    failure_reason: str | None
    mining_fingerprint: str


@dataclass(frozen=True)
class BankBuild:
    records: tuple[BankRecord, ...]
    rejected: tuple[dict[str, str], ...]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finalize_numeric_bank(
    proposals: Sequence[NeighborProposal],
    *,
    text_by_sample: Mapping[str, str],
    split_by_sample: Mapping[str, str],
    unit_mapping_hashes: Mapping[str, str],
    audit: SchemaAudit,
    mining_fingerprint: str,
) -> BankBuild:
    if not mining_fingerprint:
        raise BankError("mining fingerprint cannot be empty")
    records: list[BankRecord] = []
    rejected: list[dict[str, str]] = []
    for proposal in proposals:
        endpoints = (proposal.sample_i, proposal.sample_j)
        missing = [sample for sample in endpoints if sample not in text_by_sample]
        if missing:
            raise BankError(f"proposal references missing captions: {missing}")
        if any(split_by_sample.get(sample) != "train" for sample in endpoints):
            raise BankError(f"contrast endpoints must both be train: {proposal.pair_id}")
        if any(not unit_mapping_hashes.get(sample) for sample in endpoints):
            raise BankError(f"contrast missing unit mapping hash: {proposal.pair_id}")
        result = validate_strict_numeric_pair(
            text_by_sample[proposal.sample_i], text_by_sample[proposal.sample_j], audit
        )
        if not result.eligible or result.g_sem <= 0:
            rejected.append({"pair_id": proposal.pair_id, "reason": result.reason})
            continue
        assert result.unit_i is not None and result.unit_j is not None
        assert result.category is not None and audit.artifact_hash is not None
        records.append(
            BankRecord(
                schema_version="contrast.v1",
                pair_id=proposal.pair_id,
                sample_i=proposal.sample_i,
                sample_j=proposal.sample_j,
                text_hash_i=_text_hash(text_by_sample[proposal.sample_i]),
                text_hash_j=_text_hash(text_by_sample[proposal.sample_j]),
                unit_i=result.unit_i,
                unit_j=result.unit_j,
                unit_mapping_hash_i=unit_mapping_hashes[proposal.sample_i],
                unit_mapping_hash_j=unit_mapping_hashes[proposal.sample_j],
                category=result.category,
                schema_id=result.schema_version,
                audit_hash=audit.artifact_hash,
                g_sem=result.g_sem,
                s0_quartet=proposal.s0_quartet,
                margins0=proposal.margins0,
                hardness=proposal.hardness,
                reference_hash=None,
                grid_i=None,
                grid_j=None,
                views_i=(),
                views_j=(),
                q_i=None,
                q_j=None,
                h=None,
                stability_i=None,
                stability_j=None,
                g=0.0,
                retained_mass_i=None,
                retained_mass_j=None,
                rf_union_ratio_i=None,
                rf_union_ratio_j=None,
                support_status="pending_support",
                failure_reason=None,
                mining_fingerprint=mining_fingerprint,
            )
        )
    return BankBuild(tuple(records), tuple(rejected))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_bank(records: Sequence[BankRecord], output_dir: str | Path, fingerprint: str) -> Path:
    if any(record.mining_fingerprint != fingerprint for record in records):
        raise BankError("record fingerprint differs from requested bank fingerprint")
    if len({record.pair_id for record in records}) != len(records):
        raise BankError("bank pair IDs must be unique")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "bank.jsonl"
    temporary = output / "bank.jsonl.tmp"
    payload = "".join(_canonical_json(asdict(record)) + "\n" for record in records)
    temporary.write_text(payload, encoding="utf-8")
    checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(bank_path)
    metadata = {
        "schema_version": "bank_meta.v1",
        "mining_fingerprint": fingerprint,
        "record_count": len(records),
        "bank_sha256": checksum,
    }
    metadata_path = output / "bank_meta.json"
    metadata_tmp = output / "bank_meta.json.tmp"
    metadata_tmp.write_text(_canonical_json(metadata) + "\n", encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    return bank_path


def load_bank(output_dir: str | Path, expected_fingerprint: str) -> tuple[dict[str, Any], ...]:
    output = Path(output_dir)
    bank_path = output / "bank.jsonl"
    metadata_path = output / "bank_meta.json"
    if not bank_path.is_file() or not metadata_path.is_file():
        raise BankError(f"incomplete bank artifact: {output}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "bank_meta.v1":
        raise BankError("unsupported bank metadata schema")
    if metadata.get("mining_fingerprint") != expected_fingerprint:
        raise BankError("CACHE_HASH_MISMATCH: contrast bank fingerprint differs")
    checksum = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    if checksum != metadata.get("bank_sha256"):
        raise BankError("contrast bank checksum mismatch")
    records = tuple(json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines())
    if len(records) != metadata.get("record_count"):
        raise BankError("contrast bank record count mismatch")
    pair_ids = [record.get("pair_id") for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise BankError("contrast bank contains duplicate pair IDs")
    if any(record.get("mining_fingerprint") != expected_fingerprint for record in records):
        raise BankError("contrast record fingerprint mismatch")
    return records


def attach_pair_support(
    record: BankRecord,
    support: PairSupport,
    *,
    reference_hash: str,
    grid_i: str,
    grid_j: str,
    views_i: Sequence[str],
    views_j: Sequence[str],
) -> BankRecord:
    if record.support_status != "pending_support":
        raise BankError(f"support already resolved for pair {record.pair_id}")
    accepted = support.reason == "accepted"
    endpoint_i = support.endpoint_i
    endpoint_j = support.endpoint_j
    if accepted and (endpoint_i.q is None or endpoint_j.q is None or support.reliability <= 0):
        raise BankError("accepted pair support requires both q distributions and positive reliability")
    return replace(
        record,
        reference_hash=reference_hash,
        grid_i=grid_i,
        grid_j=grid_j,
        views_i=tuple(views_i),
        views_j=tuple(views_j),
        q_i=None if endpoint_i.q is None else tuple(map(float, endpoint_i.q)),
        q_j=None if endpoint_j.q is None else tuple(map(float, endpoint_j.q)),
        h=endpoint_i.text_distance,
        stability_i=endpoint_i.stability,
        stability_j=endpoint_j.stability,
        g=float(support.reliability) if accepted else 0.0,
        retained_mass_i=endpoint_i.retained_mass,
        retained_mass_j=endpoint_j.retained_mass,
        rf_union_ratio_i=endpoint_i.rf_union_ratio,
        rf_union_ratio_j=endpoint_j.rf_union_ratio,
        support_status="accepted" if accepted else "failed",
        failure_reason=None if accepted else support.reason,
    )
