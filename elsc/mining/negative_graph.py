from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F

from elsc.utils import stable_seed


@dataclass(frozen=True)
class PrototypeOccurrence:
    word_id: int
    video_id: str
    prototype: torch.Tensor


def raw_support_prototype(h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if h.ndim != 2 or weights.shape != h.shape[:1]:
        raise ValueError("h must be [support,D] and weights [support]")
    if not torch.isclose(weights.float().sum(), torch.tensor(1.0), atol=1e-5):
        raise ValueError("support weights must sum to one")
    normalized = F.normalize(h.float(), dim=-1)
    return F.normalize((weights.detach().float().unsqueeze(-1) * normalized).sum(0), dim=0)


def _pair_score(left: list[PrototypeOccurrence], right: list[PrototypeOccurrence]) -> float | None:
    candidates: list[tuple[float, str, str]] = []
    for lhs in left:
        for rhs in right:
            if lhs.video_id == rhs.video_id:
                continue
            score = float(torch.dot(lhs.prototype.float(), rhs.prototype.float()).item())
            candidates.append((score, lhs.video_id, rhs.video_id))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    chosen: list[float] = []
    used_left: set[str] = set()
    used_right: set[str] = set()
    for score, left_video, right_video in candidates:
        if left_video in used_left or right_video in used_right:
            continue
        chosen.append(score)
        used_left.add(left_video)
        used_right.add(right_video)
        if len(chosen) == 3:
            break
    return sum(chosen) / 3.0 if len(chosen) == 3 else None


def build_visual_neighbor_graph(
    occurrences: Iterable[PrototypeOccurrence],
    *,
    cosine_min: float = 0.70,
    top_k: int = 10,
    mutual: bool = True,
) -> dict[int, list[tuple[int, float]]]:
    by_word: dict[int, list[PrototypeOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        if occurrence.prototype.ndim != 1:
            raise ValueError("each prototype must be a vector")
        by_word[occurrence.word_id].append(occurrence)
    words = sorted(by_word)
    scores: dict[tuple[int, int], float] = {}
    for offset, left_word in enumerate(words):
        for right_word in words[offset + 1 :]:
            score = _pair_score(by_word[left_word], by_word[right_word])
            if score is not None and score >= cosine_min:
                scores[left_word, right_word] = score
                scores[right_word, left_word] = score
    directed: dict[int, list[tuple[int, float]]] = {}
    for word in words:
        neighbors = [(other, score) for (source, other), score in scores.items() if source == word]
        neighbors.sort(key=lambda value: (-value[1], value[0]))
        directed[word] = neighbors[:top_k]
    if not mutual:
        return directed
    neighbor_sets = {word: {other for other, _ in values} for word, values in directed.items()}
    return {
        word: [(other, score) for other, score in values if word in neighbor_sets.get(other, set())]
        for word, values in directed.items()
    }


def filter_occurrence_negatives(
    target_word_id: int,
    target_surface: str,
    caption_surfaces: set[str],
    graph: dict[int, list[tuple[int, float]]],
    vocabulary: dict[int, str],
    *,
    max_negatives: int = 5,
    equivalence: dict[int, set[int]] | None = None,
) -> list[int]:
    result: list[int] = []
    equivalent = (equivalence or {}).get(target_word_id, set())
    for candidate, _ in graph.get(target_word_id, []):
        surface = vocabulary[candidate].lower()
        if candidate == target_word_id or surface == target_surface.lower():
            continue
        if surface in {item.lower() for item in caption_surfaces} or candidate in equivalent:
            continue
        result.append(candidate)
        if len(result) == max_negatives:
            break
    return result


def build_frequency_matched_random_graph(
    word_counts: dict[int, int], *, seed: int, top_k: int = 10
) -> dict[int, list[tuple[int, float]]]:
    """Deterministic random control within log2 occurrence-frequency bins."""
    bins: dict[int, list[int]] = defaultdict(list)
    for word_id, count in word_counts.items():
        frequency_bin = max(0, int(count).bit_length() - 1)
        bins[frequency_bin].append(word_id)
    graph: dict[int, list[tuple[int, float]]] = {}
    for frequency_bin, words in bins.items():
        words = sorted(words)
        for word in words:
            candidates = [other for other in words if other != word]
            candidates.sort(
                key=lambda other: stable_seed(seed, frequency_bin, word, other, "random_neighbor")
            )
            graph[word] = [(other, 0.0) for other in candidates[:top_k]]
    return graph
