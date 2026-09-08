from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .neighbors import NeighborError, NeighborProposal, _margins, _pair_id


_SCHEMA = "neighbor_proposal.v1"
_META_SCHEMA = "neighbor_proposals_meta.v1"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _validated(proposal: NeighborProposal) -> NeighborProposal:
    if (
        not proposal.sample_i
        or not proposal.sample_j
        or proposal.sample_i >= proposal.sample_j
        or proposal.pair_id != _pair_id(proposal.sample_i, proposal.sample_j)
    ):
        raise NeighborError("neighbor proposal endpoints/pair ID are invalid")
    if not proposal.nominated_by or len(set(proposal.nominated_by)) != len(proposal.nominated_by):
        raise NeighborError("neighbor proposal nominations must be unique and nonempty")
    if tuple(sorted(proposal.nominated_by)) != proposal.nominated_by:
        raise NeighborError("neighbor proposal nominations must use deterministic ordering")
    if (
        not proposal.directions
        or tuple(sorted(proposal.directions)) != proposal.directions
        or not set(proposal.directions) <= {"t2v", "v2t"}
    ):
        raise NeighborError("neighbor proposal directions are invalid")
    if len(proposal.s0_quartet) != 2 or any(len(row) != 2 for row in proposal.s0_quartet):
        raise NeighborError("neighbor proposal S0 quartet must have shape [2,2]")
    values = [value for row in proposal.s0_quartet for value in row]
    values.extend(proposal.margins0)
    values.append(proposal.hardness)
    if len(proposal.margins0) != 4 or any(not math.isfinite(value) for value in values):
        raise NeighborError("neighbor proposal scores/margins must be finite")
    expected = _margins(np.asarray(proposal.s0_quartet, dtype=float))
    if any(
        abs(left - right) > 1e-7 for left, right in zip(expected, proposal.margins0, strict=True)
    ):
        raise NeighborError("neighbor proposal margins differ from its S0 quartet")
    if abs(proposal.hardness + min(expected)) > 1e-7:
        raise NeighborError("neighbor proposal hardness differs from its four margins")
    return proposal


def proposal_to_dict(proposal: NeighborProposal) -> dict[str, Any]:
    value = asdict(_validated(proposal))
    value["schema_version"] = _SCHEMA
    return value


def proposal_from_dict(value: Mapping[str, Any]) -> NeighborProposal:
    expected = {
        "schema_version",
        "pair_id",
        "sample_i",
        "sample_j",
        "nominated_by",
        "directions",
        "s0_quartet",
        "margins0",
        "hardness",
    }
    if set(value) != expected or value.get("schema_version") != _SCHEMA:
        raise NeighborError("neighbor proposal schema/fields are invalid")
    try:
        proposal = NeighborProposal(
            pair_id=str(value["pair_id"]),
            sample_i=str(value["sample_i"]),
            sample_j=str(value["sample_j"]),
            nominated_by=tuple(map(str, value["nominated_by"])),
            directions=tuple(map(str, value["directions"])),
            s0_quartet=tuple(tuple(float(item) for item in row) for row in value["s0_quartet"]),
            margins0=tuple(float(item) for item in value["margins0"]),
            hardness=float(value["hardness"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NeighborError("neighbor proposal values are invalid") from exc
    return _validated(proposal)


def write_proposals(
    proposals: Sequence[NeighborProposal],
    output_dir: str | Path,
    *,
    mining_fingerprint: str,
) -> tuple[Path, Path]:
    if not mining_fingerprint:
        raise NeighborError("proposal mining fingerprint cannot be empty")
    ordered = tuple(sorted((_validated(item) for item in proposals), key=lambda item: item.pair_id))
    if len({item.pair_id for item in ordered}) != len(ordered):
        raise NeighborError("neighbor proposal pair IDs must be unique")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    proposal_path = output / "proposals.jsonl"
    temporary = proposal_path.with_suffix(proposal_path.suffix + ".tmp")
    payload = "".join(_canonical_json(proposal_to_dict(item)) + "\n" for item in ordered)
    temporary.write_text(payload, encoding="utf-8")
    checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
    os.replace(temporary, proposal_path)
    metadata = {
        "schema_version": _META_SCHEMA,
        "mining_fingerprint": mining_fingerprint,
        "proposal_count": len(ordered),
        "proposals_sha256": checksum,
    }
    metadata_path = output / "proposals_meta.json"
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(_canonical_json(metadata) + "\n", encoding="utf-8")
    os.replace(metadata_tmp, metadata_path)
    return proposal_path, metadata_path


def load_proposals(
    output_dir: str | Path,
    *,
    expected_fingerprint: str,
    expected_sample_ids: Sequence[str] | None = None,
) -> tuple[NeighborProposal, ...]:
    output = Path(output_dir)
    proposal_path = output / "proposals.jsonl"
    metadata_path = output / "proposals_meta.json"
    if not proposal_path.is_file() or not metadata_path.is_file():
        raise NeighborError(f"incomplete neighbor proposal artifact: {output}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NeighborError("neighbor proposal metadata is unreadable") from exc
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != _META_SCHEMA
        or metadata.get("mining_fingerprint") != expected_fingerprint
    ):
        raise NeighborError("CACHE_HASH_MISMATCH: neighbor proposal provenance differs")
    if hashlib.sha256(proposal_path.read_bytes()).hexdigest() != metadata.get("proposals_sha256"):
        raise NeighborError("neighbor proposal checksum mismatch")
    proposals: list[NeighborProposal] = []
    for line_number, line in enumerate(proposal_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NeighborError(f"invalid proposal JSON at line {line_number}") from exc
        if not isinstance(value, Mapping):
            raise NeighborError(f"proposal at line {line_number} must be an object")
        proposals.append(proposal_from_dict(value))
    if len(proposals) != metadata.get("proposal_count"):
        raise NeighborError("neighbor proposal count differs from metadata")
    if [item.pair_id for item in proposals] != sorted(item.pair_id for item in proposals):
        raise NeighborError("neighbor proposals are not deterministically ordered")
    allowed = None if expected_sample_ids is None else set(map(str, expected_sample_ids))
    if allowed is not None and any(
        item.sample_i not in allowed or item.sample_j not in allowed for item in proposals
    ):
        raise NeighborError("neighbor proposal references a sample outside the expected split")
    return tuple(proposals)
