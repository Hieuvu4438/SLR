from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


class RelationError(ValueError):
    """Known positives or excluded negatives violate retrieval semantics."""


@dataclass(frozen=True)
class PairRelations:
    positives: Tensor
    candidates: Tensor
    video_rows: dict[str, int]
    text_columns: dict[str, int]


def build_pair_relations(
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    positives_by_video: Mapping[str, Iterable[str]],
    excluded_pairs: Iterable[tuple[str, str]] = (),
) -> PairRelations:
    videos = tuple(map(str, video_ids))
    texts = tuple(map(str, text_ids))
    if len(set(videos)) != len(videos) or len(set(texts)) != len(texts):
        raise RelationError("batch video/text IDs must be unique")
    video_rows = {item: index for index, item in enumerate(videos)}
    text_columns = {item: index for index, item in enumerate(texts)}
    positives = torch.zeros((len(videos), len(texts)), dtype=torch.bool)
    for video_id, target_ids in positives_by_video.items():
        if video_id not in video_rows:
            continue
        for text_id in target_ids:
            if text_id in text_columns:
                positives[video_rows[video_id], text_columns[text_id]] = True
    candidates = torch.ones_like(positives)
    for video_id, text_id in excluded_pairs:
        if video_id in video_rows and text_id in text_columns:
            candidates[video_rows[video_id], text_columns[text_id]] = False
    if bool((positives & ~candidates).any()):
        raise RelationError("a known positive cannot be excluded from candidates")
    if len(videos) and not bool(positives.any(dim=1).all()):
        raise RelationError("every video query in a training batch needs a known positive")
    if len(texts) and not bool(positives.any(dim=0).all()):
        raise RelationError("every text query in a training batch needs a known positive")
    return PairRelations(positives, candidates, video_rows, text_columns)
