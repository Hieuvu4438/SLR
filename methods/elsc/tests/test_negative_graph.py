from __future__ import annotations

import pytest
import torch

from elsc.mining.negative_graph import (
    PrototypeOccurrence,
    _batched_pair_scores,
    _pair_score,
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


def test_batched_pair_scores_match_scalar_greedy_reference():
    generator = torch.Generator().manual_seed(7)
    by_word = {}
    for word_id in range(5):
        values = []
        for index in range(3 + word_id):
            prototype = torch.nn.functional.normalize(
                torch.randn(8, generator=generator), dim=0
            )
            # Shared IDs exercise the cross-word exclusion; IDs remain unique
            # within a word, as guaranteed by the capped mining input.
            values.append(PrototypeOccurrence(word_id, f"video-{index}", prototype))
        by_word[word_id] = values
    expected = {}
    for left in range(5):
        for right in range(left + 1, 5):
            expected[left, right] = _pair_score(by_word[left], by_word[right])
    actual = _batched_pair_scores(
        by_word, device=torch.device("cpu"), pair_batch_size=3
    )
    assert set(actual) == set(expected)
    for pair, score in actual.items():
        assert score == pytest.approx(expected[pair], abs=1e-7)
