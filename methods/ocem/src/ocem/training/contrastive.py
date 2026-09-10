"""Duplicate-safe candidate construction and exact symmetric contrastive loss."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch


class ContrastiveContractError(ValueError):
    """Raised when a batch would change positives or loss denominators."""


@dataclass(frozen=True)
class CandidateSets:
    row_candidates: tuple[tuple[int, ...], ...]
    column_candidates: tuple[tuple[int, ...], ...]
    pair_union: tuple[tuple[int, int], ...]
    row_hard_counts: tuple[int, ...]
    row_random_counts: tuple[int, ...]
    column_hard_counts: tuple[int, ...]
    column_random_counts: tuple[int, ...]


def exact_duplicate_negative_mask(
    sample_ids: Sequence[str], caption_hashes: Sequence[str]
) -> torch.Tensor:
    """Return ``M[i,k] = (i==k or caption_hash[i]!=caption_hash[k])``."""

    if not sample_ids or len(sample_ids) != len(caption_hashes):
        raise ContrastiveContractError("sample IDs and caption hashes must have equal nonzero length")
    if len(set(sample_ids)) != len(sample_ids):
        raise ContrastiveContractError("repeated sample IDs must be fixed by the sampler")
    hashes = list(map(str, caption_hashes))
    size = len(hashes)
    mask = torch.empty((size, size), dtype=torch.bool)
    for row in range(size):
        for column in range(size):
            mask[row, column] = row == column or hashes[row] != hashes[column]
    return mask


def _normalize_candidates(
    candidates: Sequence[Sequence[int]], allowed: torch.Tensor, *, axis: str
) -> tuple[tuple[int, ...], ...]:
    size = allowed.shape[0]
    if len(candidates) != size:
        raise ContrastiveContractError(f"{axis} candidate list must have {size} anchors")
    normalized = []
    for anchor, values in enumerate(candidates):
        unique = tuple(dict.fromkeys(int(value) for value in values))
        if anchor not in unique:
            raise ContrastiveContractError(f"{axis} candidates for anchor {anchor} omit its positive")
        if len(unique) < 2:
            raise ContrastiveContractError(f"{axis} anchor {anchor} has no eligible negative")
        for candidate in unique:
            if not 0 <= candidate < size:
                raise ContrastiveContractError(f"{axis} candidate index is out of range")
            permitted = allowed[anchor, candidate] if axis == "row" else allowed[candidate, anchor]
            if not bool(permitted):
                raise ContrastiveContractError(f"{axis} candidates include a duplicate-caption negative")
        normalized.append(unique)
    return tuple(normalized)


def symmetric_candidate_loss(
    logits: torch.Tensor,
    *,
    allowed_mask: torch.Tensor,
    row_candidates: Sequence[Sequence[int]] | None = None,
    column_candidates: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Compute one global symmetric loss without adding another temperature."""

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ContrastiveContractError("logits must be a square [video,text] matrix")
    size = logits.shape[0]
    if allowed_mask.shape != logits.shape or allowed_mask.dtype is not torch.bool:
        raise ContrastiveContractError("allowed_mask must be a bool matrix matching logits")
    default_rows = [torch.nonzero(allowed_mask[row], as_tuple=False).flatten().tolist() for row in range(size)]
    default_columns = [
        torch.nonzero(allowed_mask[:, column], as_tuple=False).flatten().tolist()
        for column in range(size)
    ]
    rows = _normalize_candidates(row_candidates or default_rows, allowed_mask, axis="row")
    columns = _normalize_candidates(
        column_candidates or default_columns, allowed_mask, axis="column"
    )
    row_losses = [
        torch.logsumexp(logits[row, list(candidates)], dim=0) - logits[row, row]
        for row, candidates in enumerate(rows)
    ]
    column_losses = [
        torch.logsumexp(logits[list(candidates), column], dim=0) - logits[column, column]
        for column, candidates in enumerate(columns)
    ]
    return (torch.stack(row_losses).mean() + torch.stack(column_losses).mean()) / 2


def _anchor_rng(seed: int, direction: str, anchor_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{direction}\0{anchor_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _select_one(
    scores: torch.Tensor,
    eligible: Sequence[int],
    candidate_ids: Sequence[str],
    *,
    max_hard: int,
    max_random: int,
    rng: random.Random,
) -> tuple[list[int], int, int]:
    ordered = sorted(eligible, key=lambda index: (-float(scores[index]), candidate_ids[index]))
    hard = ordered[:max_hard]
    remainder = sorted(set(eligible).difference(hard), key=lambda index: candidate_ids[index])
    random_count = min(max_random, len(remainder))
    sampled = rng.sample(remainder, random_count)
    return hard + sampled, len(hard), len(sampled)


def select_mixed_candidates(
    miner_v2t: torch.Tensor,
    miner_t2v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor,
    sample_ids: Sequence[str],
    seed: int,
    max_hard: int = 8,
    max_random: int = 8,
) -> CandidateSets:
    """Build separate row/column pools; their union is compute-only."""

    if miner_v2t.shape != miner_t2v.shape or miner_v2t.ndim != 2:
        raise ContrastiveContractError("miner matrices must have one matching square shape")
    size = miner_v2t.shape[0]
    if miner_v2t.shape[1] != size or allowed_mask.shape != miner_v2t.shape:
        raise ContrastiveContractError("miner and allowed-mask shapes must be square and equal")
    if allowed_mask.dtype is not torch.bool or len(sample_ids) != size:
        raise ContrastiveContractError("sample IDs/mask do not match miner matrices")
    if len(set(sample_ids)) != size or max_hard < 0 or max_random < 0:
        raise ContrastiveContractError("sample IDs must be unique and candidate budgets nonnegative")

    rows, columns = [], []
    row_hard, row_random, column_hard, column_random = [], [], [], []
    for anchor in range(size):
        row_eligible = [
            index for index in range(size) if index != anchor and bool(allowed_mask[anchor, index])
        ]
        column_eligible = [
            index for index in range(size) if index != anchor and bool(allowed_mask[index, anchor])
        ]
        if not row_eligible or not column_eligible:
            raise ContrastiveContractError(
                f"anchor {sample_ids[anchor]!r} has no eligible real negative"
            )
        row_selected, hard_count, random_count = _select_one(
            miner_v2t[anchor],
            row_eligible,
            sample_ids,
            max_hard=max_hard,
            max_random=max_random,
            rng=_anchor_rng(seed, "row_v2t", sample_ids[anchor]),
        )
        rows.append((anchor, *row_selected))
        row_hard.append(hard_count)
        row_random.append(random_count)
        column_selected, hard_count, random_count = _select_one(
            miner_t2v[:, anchor],
            column_eligible,
            sample_ids,
            max_hard=max_hard,
            max_random=max_random,
            rng=_anchor_rng(seed, "column_t2v", sample_ids[anchor]),
        )
        columns.append((anchor, *column_selected))
        column_hard.append(hard_count)
        column_random.append(random_count)
    pair_union = {(row, column) for row, values in enumerate(rows) for column in values}
    pair_union.update(
        (row, column) for column, values in enumerate(columns) for row in values
    )
    return CandidateSets(
        row_candidates=tuple(rows),
        column_candidates=tuple(columns),
        pair_union=tuple(sorted(pair_union)),
        row_hard_counts=tuple(row_hard),
        row_random_counts=tuple(row_random),
        column_hard_counts=tuple(column_hard),
        column_random_counts=tuple(column_random),
    )


def score_pair_union_in_blocks(
    scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pairs: Sequence[tuple[int, int]],
    *,
    block_size: int,
    device: torch.device | str,
) -> dict[tuple[int, int], torch.Tensor]:
    """Score each selected pair once while retaining its autograd graph."""

    unique = tuple(dict.fromkeys((int(row), int(column)) for row, column in pairs))
    if not unique or block_size < 1:
        raise ContrastiveContractError("pair union must be non-empty and block_size positive")
    result: dict[tuple[int, int], torch.Tensor] = {}
    for start in range(0, len(unique), block_size):
        block = unique[start : start + block_size]
        rows = torch.tensor([pair[0] for pair in block], dtype=torch.long, device=device)
        columns = torch.tensor([pair[1] for pair in block], dtype=torch.long, device=device)
        scores = scorer(rows, columns)
        if scores.ndim != 1 or scores.shape[0] != len(block):
            raise ContrastiveContractError("pair scorer must return one scalar per pair")
        result.update({pair: scores[index] for index, pair in enumerate(block)})
    return result


def symmetric_loss_from_pair_scores(
    pair_scores: Mapping[tuple[int, int], torch.Tensor],
    *,
    row_candidates: Sequence[Sequence[int]],
    column_candidates: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Assemble exact row/column denominators from a compute-deduplicated union."""

    size = len(row_candidates)
    if len(column_candidates) != size:
        raise ContrastiveContractError("row and column candidate anchors differ")
    losses_row, losses_column = [], []
    for anchor in range(size):
        positive = pair_scores.get((anchor, anchor))
        if positive is None:
            raise ContrastiveContractError(f"positive pair ({anchor},{anchor}) is missing")
        try:
            row_logits = torch.stack([pair_scores[(anchor, index)] for index in row_candidates[anchor]])
            column_logits = torch.stack(
                [pair_scores[(index, anchor)] for index in column_candidates[anchor]]
            )
        except KeyError as error:
            raise ContrastiveContractError(f"selected pair is missing from compute union: {error}") from error
        losses_row.append(torch.logsumexp(row_logits, dim=0) - positive)
        losses_column.append(torch.logsumexp(column_logits, dim=0) - positive)
    return (torch.stack(losses_row).mean() + torch.stack(losses_column).mean()) / 2
