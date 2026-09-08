from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

from .neighbors import NeighborProposal


def export_audit_template(
    proposals: Sequence[NeighborProposal],
    captions: Mapping[str, str],
    output: str | Path,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, object]:
    """Export blinded human-rating rows; never infer or prefill semantic judgments."""
    if sample_size <= 0:
        raise ValueError("audit sample_size must be positive")
    ordered = sorted(proposals, key=lambda item: item.pair_id)
    randomizer = random.Random(seed)
    selected = ordered if len(ordered) <= sample_size else randomizer.sample(ordered, sample_size)
    selected = sorted(selected, key=lambda item: item.pair_id)
    rows: list[dict[str, object]] = []
    for proposal in selected:
        if proposal.sample_i not in captions or proposal.sample_j not in captions:
            raise ValueError(f"missing audit caption for {proposal.pair_id}")
        rows.append(
            {
                "schema_version": "contrast_audit_row.v1",
                "pair_id": proposal.pair_id,
                "sample_i": proposal.sample_i,
                "sample_j": proposal.sample_j,
                "text_i": captions[proposal.sample_i],
                "text_j": captions[proposal.sample_j],
                "category": None,
                "positive_i_rating": None,
                "positive_j_rating": None,
                "cross_i_j_negative_rating": None,
                "cross_j_i_negative_rating": None,
                "support_i_correct": None,
                "support_j_correct": None,
                "uncertain": None,
                "rater_id": None,
                "notes": None,
            }
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": "contrast_audit_export.v1",
        "seed": seed,
        "requested_size": sample_size,
        "exported_size": len(rows),
        "proposal_count": len(proposals),
        "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "pair_ids": [row["pair_id"] for row in rows],
    }
