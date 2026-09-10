"""Post-hoc full-gallery error and exact-caption diagnostics."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


class ErrorSelectionError(ValueError):
    """Raised when a full-gallery score/ID contract is invalid."""


def _order(scores: np.ndarray, candidate_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(list(map(str, candidate_ids)))
    if values.shape != (len(ids),) or not np.isfinite(values).all():
        raise ErrorSelectionError("candidate scores must be one finite vector")
    return np.lexsort((ids, -values))


def select_r1_errors(
    scores: np.ndarray,
    *,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    direction: str,
) -> list[dict[str, object]]:
    """Select each true pair and highest-ranked wrong pair using canonical ties."""

    matrix = np.asarray(scores, dtype=np.float64)
    videos, texts = list(map(str, video_ids)), list(map(str, text_ids))
    if matrix.shape != (len(videos), len(texts)) or not np.isfinite(matrix).all():
        raise ErrorSelectionError("score matrix/IDs are not a finite full gallery")
    if len(set(videos)) != len(videos) or len(set(texts)) != len(texts):
        raise ErrorSelectionError("gallery IDs must be unique")
    video_pos, text_pos = {value: i for i, value in enumerate(videos)}, {
        value: i for i, value in enumerate(texts)
    }
    if set(video_pos) != set(text_pos):
        raise ErrorSelectionError("paired diagnostic galleries must have identical ID sets")
    errors = []
    if direction == "T2V":
        for text_index, query_id in enumerate(texts):
            target = video_pos[query_id]
            ordered = _order(matrix[:, text_index], videos)
            rank = int(np.flatnonzero(ordered == target)[0]) + 1
            if rank > 1:
                false = int(ordered[0])
                errors.append(
                    {
                        "direction": direction,
                        "query_id": query_id,
                        "true_video_id": videos[target],
                        "true_text_id": query_id,
                        "false_video_id": videos[false],
                        "false_text_id": query_id,
                        "true_rank": rank,
                        "true_score": float(matrix[target, text_index]),
                        "false_score": float(matrix[false, text_index]),
                    }
                )
    elif direction == "V2T":
        for video_index, query_id in enumerate(videos):
            target = text_pos[query_id]
            ordered = _order(matrix[video_index], texts)
            rank = int(np.flatnonzero(ordered == target)[0]) + 1
            if rank > 1:
                false = int(ordered[0])
                errors.append(
                    {
                        "direction": direction,
                        "query_id": query_id,
                        "true_video_id": query_id,
                        "true_text_id": texts[target],
                        "false_video_id": query_id,
                        "false_text_id": texts[false],
                        "true_rank": rank,
                        "true_score": float(matrix[video_index, target]),
                        "false_score": float(matrix[video_index, false]),
                    }
                )
    else:
        raise ErrorSelectionError("direction must be T2V or V2T")
    return errors


def equivalence_recall(
    scores: np.ndarray,
    *,
    video_ids: Sequence[str],
    text_ids: Sequence[str],
    caption_hash_by_id: Mapping[str, str],
    direction: str,
) -> dict[str, float | int]:
    """Report exact-caption equivalence recall without changing paired-ID recall."""

    matrix = np.asarray(scores, dtype=np.float64)
    videos, texts = list(map(str, video_ids)), list(map(str, text_ids))
    if set(caption_hash_by_id) != set(videos) or set(videos) != set(texts):
        raise ErrorSelectionError("caption classes do not align with paired galleries")
    ranks = []
    query_ids = texts if direction == "T2V" else videos
    candidates = videos if direction == "T2V" else texts
    for query_index, query_id in enumerate(query_ids):
        values = matrix[:, query_index] if direction == "T2V" else matrix[query_index]
        ordered = _order(values, candidates)
        target_hash = caption_hash_by_id[query_id]
        rank = next(
            position + 1
            for position, candidate_index in enumerate(ordered)
            if caption_hash_by_id[candidates[int(candidate_index)]] == target_hash
        )
        ranks.append(rank)
    values = np.asarray(ranks)
    return {
        "queries": len(ranks),
        "R1": float(100.0 * np.mean(values <= 1)),
        "R5": float(100.0 * np.mean(values <= 5)),
        "R10": float(100.0 * np.mean(values <= 10)),
    }
