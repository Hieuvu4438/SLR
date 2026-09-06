from __future__ import annotations

import torch

from elsc.mining.negative_graph import (
    PrototypeOccurrence,
    build_frequency_matched_random_graph,
    build_visual_neighbor_graph,
    filter_occurrence_negatives,
)


def test_visual_graph_requires_three_distinct_video_pairs_and_mutual_neighbors():
    occurrences = []
    for word_id in (0, 1):
        for index in range(3):
            occurrences.append(
                PrototypeOccurrence(word_id, f"w{word_id}-v{index}", torch.tensor([1.0, 0.0]))
            )
    graph = build_visual_neighbor_graph(occurrences, cosine_min=0.9, top_k=10, mutual=True)
    assert graph == {0: [(1, 1.0)], 1: [(0, 1.0)]}
    insufficient = build_visual_neighbor_graph(occurrences[:2] + occurrences[3:5], cosine_min=0.0)
    assert insufficient == {0: [], 1: []}


def test_occurrence_filters_caption_words_and_equivalence():
    graph = {0: [(1, 0.9), (2, 0.8), (3, 0.7)]}
    vocabulary = {0: "rain", 1: "snow", 2: "wind", 3: "sun"}
    result = filter_occurrence_negatives(
        0, "rain", {"rain", "wind"}, graph, vocabulary, equivalence={0: {3}}
    )
    assert result == [1]


def test_random_control_is_seeded_and_frequency_matched():
    counts = {0: 4, 1: 5, 2: 8, 3: 9}
    first = build_frequency_matched_random_graph(counts, seed=42, top_k=2)
    second = build_frequency_matched_random_graph(counts, seed=42, top_k=2)
    assert first == second
    assert {neighbor for neighbor, _ in first[0]} == {1}
    assert {neighbor for neighbor, _ in first[2]} == {3}
