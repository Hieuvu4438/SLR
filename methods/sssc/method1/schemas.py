from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


SCHEMA_VERSION = 1


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class TextRecord:
    text_uid: str
    dataset: str
    split: str
    group_uid: str
    official_order: int
    raw_text: str
    canonical_text: str
    caption_hash: str
    language: str
    original_language_text: str | None = None

    def validate(self) -> None:
        if self.split not in {"train", "dev", "test"}:
            raise SchemaError(f"invalid text split: {self.split}")
        if not all((self.text_uid, self.dataset, self.group_uid, self.caption_hash, self.language)):
            raise SchemaError("text identity fields must be non-empty")
        if self.official_order < 0:
            raise SchemaError("official_order must be non-negative")


@dataclass(frozen=True)
class VideoRecord:
    video_uid: str
    dataset: str
    split: str
    group_uid: str
    official_order: int
    agnostic_path: str
    aware_path: str
    agnostic_sha256: str
    aware_sha256: str
    num_feature_rows: int
    signer_id: str | None = None

    def validate(self) -> None:
        if self.split not in {"train", "dev", "test"}:
            raise SchemaError(f"invalid video split: {self.split}")
        if not all((self.video_uid, self.dataset, self.group_uid, self.agnostic_path, self.aware_path)):
            raise SchemaError("video identity/path fields must be non-empty")
        if self.official_order < 0 or self.num_feature_rows < 1:
            raise SchemaError("video order must be non-negative and features non-empty")


@dataclass(frozen=True)
class GroupRecord:
    group_uid: str
    video_uids: tuple[str, ...]
    text_uid: str
    official_order: int

    def validate(self) -> None:
        if not self.group_uid or not self.text_uid or not self.video_uids:
            raise SchemaError("group identity and membership must be non-empty")
        if len(set(self.video_uids)) != len(self.video_uids):
            raise SchemaError(f"duplicate video in group {self.group_uid}")
        if self.official_order < 0:
            raise SchemaError("official_order must be non-negative")


@dataclass(frozen=True)
class EditRecord:
    schema_version: int
    edit_uid: str
    text_uid: str
    positive_caption_hash: str
    negative_canonical_text: str
    negative_caption_hash: str
    occurrence_uid: str
    source_word: str
    replacement_word: str
    positive_char_span: tuple[int, int]
    negative_char_span: tuple[int, int]
    positive_token_positions: tuple[int, ...]
    negative_token_positions: tuple[int, ...]
    positive_full_bpe_count: int
    negative_full_bpe_count: int
    prototype_cosine: float
    eligible: bool
    rejection_reason: str | None
    miner_version: str

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError("edit schema_version must be 1")
        if not all((self.edit_uid, self.text_uid, self.occurrence_uid, self.source_word, self.replacement_word)):
            raise SchemaError("edit identity fields must be non-empty")
        if self.source_word == self.replacement_word:
            raise SchemaError("edit replacement must differ from source")
        if self.eligible and self.rejection_reason is not None:
            raise SchemaError("eligible edit cannot carry a rejection reason")
        if not self.eligible and not self.rejection_reason:
            raise SchemaError("ineligible edit must carry a rejection reason")


@dataclass
class LocalEncoding:
    video_raw: torch.Tensor
    video_ignore_raw: torch.Tensor
    text_raw: torch.Tensor
    text_valid: torch.Tensor
    text_aug_raw: torch.Tensor
    text_aug_valid: torch.Tensor


@dataclass
class AuxiliaryTerms:
    numerator: torch.Tensor
    denominator: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int | None, ...],
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None,
    finite: bool = False,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise SchemaError(f"{name} must be a tensor")
    if tensor.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(tensor.shape, shape, strict=True)
    ):
        raise SchemaError(f"{name} expected shape {shape}, got {tuple(tensor.shape)}")
    if dtype is not None:
        accepted = dtype if isinstance(dtype, tuple) else (dtype,)
        if tensor.dtype not in accepted:
            raise SchemaError(f"{name} expected dtype {accepted}, got {tensor.dtype}")
    if finite and not bool(torch.isfinite(tensor).all()):
        raise SchemaError(f"{name} contains NaN or infinity")


def as_json_record(record: Any) -> dict[str, Any]:
    from dataclasses import asdict

    value = asdict(record)
    return value
