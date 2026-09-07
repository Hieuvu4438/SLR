from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F

from slr_common.utils import stable_seed


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


@torch.no_grad()
def _batched_pair_scores(
    by_word: dict[int, list[PrototypeOccurrence]],
    *,
    device: torch.device,
    pair_batch_size: int,
) -> dict[tuple[int, int], float]:
    """Compute the exact greedy three-pair score in padded GPU/CPU batches."""
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    words = sorted(by_word)
    if not words:
        return {}
    ordered = {
        word: sorted(by_word[word], key=lambda item: item.video_id) for word in words
    }
    dimensions = {
        int(item.prototype.numel()) for values in ordered.values() for item in values
    }
    if len(dimensions) != 1:
        raise ValueError("all occurrence prototypes must have one shared dimension")
    dimension = dimensions.pop()
    maximum = max(len(values) for values in ordered.values())
    if maximum == 0:
        return {}
    all_video_ids = sorted(
        {item.video_id for values in ordered.values() for item in values}
    )
    video_to_id = {video_id: index for index, video_id in enumerate(all_video_ids)}
    prototypes = torch.zeros(len(words), maximum, dimension, dtype=torch.float32)
    video_ids = torch.full((len(words), maximum), -1, dtype=torch.long)
    valid = torch.zeros(len(words), maximum, dtype=torch.bool)
    for word_index, word in enumerate(words):
        values = ordered[word]
        if not values:
            continue
        prototypes[word_index, : len(values)] = torch.stack(
            [item.prototype.float() for item in values]
        )
        video_ids[word_index, : len(values)] = torch.tensor(
            [video_to_id[item.video_id] for item in values]
        )
        valid[word_index, : len(values)] = True
    prototypes = prototypes.to(device)
    video_ids = video_ids.to(device)
    valid = valid.to(device)
    eligible = [index for index, word in enumerate(words) if len(ordered[word]) >= 3]
    if len(eligible) < 2:
        return {}
    pair_indexes = torch.triu_indices(len(eligible), len(eligible), offset=1)
    eligible_tensor = torch.tensor(eligible, dtype=torch.long)
    left_all = eligible_tensor.index_select(0, pair_indexes[0])
    right_all = eligible_tensor.index_select(0, pair_indexes[1])
    result: dict[tuple[int, int], float] = {}
    for offset in range(0, len(left_all), pair_batch_size):
        left_index_cpu = left_all[offset : offset + pair_batch_size]
        right_index_cpu = right_all[offset : offset + pair_batch_size]
        left_index = left_index_cpu.to(device)
        right_index = right_index_cpu.to(device)
        left_prototypes = prototypes.index_select(0, left_index)
        right_prototypes = prototypes.index_select(0, right_index)
        scores = torch.bmm(left_prototypes, right_prototypes.transpose(1, 2))
        left_videos = video_ids.index_select(0, left_index)
        right_videos = video_ids.index_select(0, right_index)
        pair_valid = (
            valid.index_select(0, left_index).unsqueeze(2)
            & valid.index_select(0, right_index).unsqueeze(1)
            & (left_videos.unsqueeze(2) != right_videos.unsqueeze(1))
        )
        scores.masked_fill_(~pair_valid, float("-inf"))
        selected = torch.zeros(len(left_index), dtype=torch.float32, device=device)
        complete = torch.ones(len(left_index), dtype=torch.bool, device=device)
        for _ in range(3):
            best, flat_index = scores.flatten(1).max(dim=1)
            finite = torch.isfinite(best)
            complete &= finite
            selected += torch.where(finite, best, torch.zeros_like(best))
            row = torch.div(flat_index, maximum, rounding_mode="floor")
            column = flat_index.remainder(maximum)
            selected_left_video = left_videos.gather(1, row[:, None])
            selected_right_video = right_videos.gather(1, column[:, None])
            used_left = left_videos.unsqueeze(2) == selected_left_video.unsqueeze(2)
            used_right = right_videos.unsqueeze(1) == selected_right_video.unsqueeze(1)
            scores.masked_fill_(used_left | used_right, float("-inf"))
        means = (selected / 3.0).cpu()
        complete_cpu = complete.cpu()
        for local_index in complete_cpu.nonzero(as_tuple=False).flatten().tolist():
            left_word = words[int(left_index_cpu[local_index])]
            right_word = words[int(right_index_cpu[local_index])]
            result[left_word, right_word] = float(means[local_index])
    return result


def build_visual_neighbor_graph(
    occurrences: Iterable[PrototypeOccurrence],
    *,
    cosine_min: float = 0.70,
    top_k: int = 10,
    mutual: bool = True,
    device: str | torch.device = "cpu",
    pair_batch_size: int = 512,
) -> dict[int, list[tuple[int, float]]]:
    by_word: dict[int, list[PrototypeOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        if occurrence.prototype.ndim != 1:
            raise ValueError("each prototype must be a vector")
        by_word[occurrence.word_id].append(occurrence)
    words = sorted(by_word)
    pair_scores = _batched_pair_scores(
        by_word, device=torch.device(device), pair_batch_size=pair_batch_size
    )
    scores: dict[tuple[int, int], float] = {}
    for (left_word, right_word), score in pair_scores.items():
        if score >= cosine_min:
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
