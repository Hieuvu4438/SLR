from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .reference import (
    ReferenceCacheIdentity,
    ReferenceCacheError,
    validate_negative_span_cache,
    validate_reference_cache,
)
from .sampling import choose_epoch_edits
from .schemas import EditRecord
from .utils import sha256_file, sha256_json


class AuxiliaryCacheError(RuntimeError):
    pass


def _read_index(
    path: Path, identity_field: str, expected_count: int
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            identity = str(record[identity_field])
            if identity in output:
                raise AuxiliaryCacheError(f"duplicate {identity_field} in {path}")
            output[identity] = record
    rows = {int(record["row"]) for record in output.values()}
    if rows != set(range(expected_count)):
        raise AuxiliaryCacheError(f"rows in {path} are not a complete stable index")
    return output


class Method1AuxiliaryCache:
    def __init__(
        self,
        cache_root: str | Path,
        *,
        expected_identity: ReferenceCacheIdentity,
    ) -> None:
        self.root = Path(cache_root)
        arrays = validate_reference_cache(self.root, expected_identity=expected_identity)
        self.video_tokens = arrays["reference_video_tokens"]
        self.positive_spans = arrays["reference_positive_spans"]
        self.video_index = _read_index(
            self.root / "video_index.jsonl", "video_uid", len(self.video_tokens)
        )
        self.positive_index = _read_index(
            self.root / "positive_span_index.jsonl",
            "occurrence_uid",
            len(self.positive_spans),
        )
        bundle_root = self.root / "auxiliary"
        mining_root = bundle_root / "mining"
        try:
            mining_report = json.loads(
                (mining_root / "mining_report.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise AuxiliaryCacheError(f"cannot read mining report: {error}") from error
        if mining_report.get("status") != "complete":
            raise AuxiliaryCacheError("mining report is not complete")
        stored_content_hash = mining_report.get("content_sha256")
        hash_payload = dict(mining_report)
        hash_payload.pop("content_sha256", None)
        if stored_content_hash != sha256_json(hash_payload):
            raise AuxiliaryCacheError("mining report content hash mismatch")
        if (
            mining_report.get("resource_hashes", {}).get("reference_identity_sha256")
            != expected_identity.digest
        ):
            raise AuxiliaryCacheError("miner was not built from the requested reference cache")
        for name, digest in mining_report.get("files", {}).items():
            if sha256_file(mining_root / name) != digest:
                raise AuxiliaryCacheError(f"mining artifact hash mismatch: {name}")
        try:
            self.negative_spans, self.negative_index = validate_negative_span_cache(
                bundle_root,
                reference_identity_sha256=expected_identity.digest,
                mining_content_sha256=stored_content_hash,
            )
        except ReferenceCacheError as error:
            raise AuxiliaryCacheError(str(error)) from error

        allowed = {field.name for field in fields(EditRecord)}
        edits_by_text: dict[str, list[EditRecord]] = {}
        seen_edits: set[str] = set()
        with (mining_root / "edits.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                unknown = set(raw) - allowed - {"caption_order"}
                if unknown:
                    raise AuxiliaryCacheError(f"unknown edit fields: {sorted(unknown)}")
                raw.pop("caption_order", None)
                for name in (
                    "positive_char_span",
                    "negative_char_span",
                    "positive_token_positions",
                    "negative_token_positions",
                ):
                    raw[name] = tuple(raw[name])
                edit = EditRecord(**raw)
                edit.validate()
                if edit.edit_uid in seen_edits or edit.edit_uid not in self.negative_index:
                    raise AuxiliaryCacheError(f"duplicate or uncached edit: {edit.edit_uid}")
                if edit.occurrence_uid not in self.positive_index:
                    raise AuxiliaryCacheError(
                        f"edit has no cached positive occurrence: {edit.edit_uid}"
                    )
                seen_edits.add(edit.edit_uid)
                edits_by_text.setdefault(edit.text_uid, []).append(edit)
        if seen_edits != set(self.negative_index):
            raise AuxiliaryCacheError("negative span cache and edit table identities differ")
        self.edits_by_text = {key: tuple(values) for key, values in edits_by_text.items()}
        self.edits_by_uid = {
            edit.edit_uid: edit for edits in self.edits_by_text.values() for edit in edits
        }
        self.reference_identity_sha256 = expected_identity.digest
        self.mining_content_sha256 = stored_content_hash

    def edit(self, edit_uid: str) -> EditRecord:
        try:
            return self.edits_by_uid[edit_uid]
        except KeyError as error:
            raise AuxiliaryCacheError(f"unknown cached edit: {edit_uid}") from error

    def training_fields(
        self,
        *,
        video_uid: str,
        text_uid: str,
        caption_hash: str,
        seed: int,
        epoch: int,
        negatives_per_caption: int,
    ) -> dict[str, Any]:
        if video_uid not in self.video_index:
            raise AuxiliaryCacheError(f"video is absent from reference cache: {video_uid}")
        candidates = self.edits_by_text.get(text_uid, ())
        if any(edit.positive_caption_hash != caption_hash for edit in candidates):
            raise AuxiliaryCacheError(f"caption hash changed after mining: {text_uid}")
        selected_uids, selected_valid = choose_epoch_edits(
            [edit.edit_uid for edit in candidates],
            k=negatives_per_caption,
            seed=seed,
            epoch=epoch,
            text_uid=text_uid,
        )
        by_uid = {edit.edit_uid: edit for edit in candidates}
        dimension = int(self.video_tokens.shape[-1])
        positive = np.zeros((negatives_per_caption, 1, dimension), dtype=np.float32)
        negative = np.zeros_like(positive)
        valid = np.zeros((negatives_per_caption, 1), dtype=np.bool_)
        confidence = np.zeros((negatives_per_caption, 1), dtype=np.float32)
        nested_uids: list[list[str | None]] = []
        for slot, edit_uid in enumerate(selected_uids):
            nested_uids.append([edit_uid])
            if edit_uid is None:
                continue
            edit = by_uid[edit_uid]
            positive_row = int(self.positive_index[edit.occurrence_uid]["row"])
            negative_row = int(self.negative_index[edit.edit_uid])
            positive[slot, 0] = self.positive_spans[positive_row]
            negative[slot, 0] = self.negative_spans[negative_row]
            valid[slot, 0] = bool(selected_valid[slot])
            confidence[slot, 0] = 1.0
        video_row = int(self.video_index[video_uid]["row"])
        return {
            "edit_uids": nested_uids,
            "x_ref": torch.from_numpy(np.array(self.video_tokens[video_row], copy=True)),
            "q_pos": torch.from_numpy(positive),
            "q_neg": torch.from_numpy(negative),
            "edit_valid": torch.from_numpy(valid),
            "confidence": torch.from_numpy(confidence),
        }
