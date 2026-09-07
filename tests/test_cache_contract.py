from __future__ import annotations

import pytest
import torch

from elsc.data.cache_dataset import (
    CacheMismatchError,
    _shuffled_lexical_id,
    lexical_tensors_from_batch,
    validate_cache_meta,
)


def _meta():
    return {
        "schema_version": 1,
        "split": "train",
        "teacher_hash": "teacher",
        "config_hash": "config",
        "manifest_hash": "manifest",
        "tokenizer_hash": "tokenizer",
        "language": "en",
        "feature_fusion": "sum:0.9",
        "view_sampling": "upstream_uniform",
        "mining_version": "v1",
        "negative_table_hash": "negative",
        "lexical_bank_hash": "bank",
    }


def test_train_only_and_exact_hash_match():
    validate_cache_meta(_meta(), {"teacher_hash": "teacher", "language": "en"})
    invalid = _meta()
    invalid["split"] = "test"
    with pytest.raises(CacheMismatchError, match="train-only"):
        validate_cache_meta(invalid, {})
    with pytest.raises(CacheMismatchError, match="teacher_hash"):
        validate_cache_meta(_meta(), {"teacher_hash": "different"})


def test_random_support_matches_duration_and_does_not_overlap_teacher():
    z = torch.tensor(
        [[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]]
    )
    dense = torch.tensor([[0, 2, 4, 6, 8, 10]])
    record = {
        "pair_id": "pair-1",
        "word_id": 0,
        "support_dense_indices": [2, 4],
        "support_weights": [0.4, 0.6],
        "negative_word_ids": [1],
        "rho": 0.8,
    }
    support_z, weights, *_ = lexical_tensors_from_batch(
        z,
        dense,
        [[record]],
        torch.eye(2),
        support_mode="random_matched",
        seed=42,
        random_span_duration_tolerance=0.10,
    )
    selected_positions = support_z[0, :, 0].long().tolist()
    assert not set(selected_positions) & {1, 2}
    selected_duration = int(dense[0, selected_positions[-1]] - dense[0, selected_positions[0]] + 1)
    assert selected_duration == 3
    assert weights[0].tolist() == pytest.approx([0.4, 0.6])


def test_random_support_abstains_without_duration_matched_disjoint_span():
    z = torch.randn(1, 5, 2)
    dense = torch.tensor([[0, 1, 10, 20, 40]])
    record = {
        "pair_id": "pair-2",
        "word_id": 0,
        "support_dense_indices": [0, 1],
        "support_weights": [0.5, 0.5],
        "negative_word_ids": [1],
        "rho": 0.8,
    }
    support_z, *_ = lexical_tensors_from_batch(
        z,
        dense,
        [[record]],
        torch.eye(2),
        support_mode="random_matched",
        random_span_duration_tolerance=0.10,
    )
    assert len(support_z) == 0


def test_shuffled_lexical_control_is_fixed_point_free_and_batch_independent():
    vocabulary_size = 11
    mapping = [
        _shuffled_lexical_id(word_id, vocabulary_size, seed=42)
        for word_id in range(vocabulary_size)
    ]
    assert sorted(mapping) == list(range(vocabulary_size))
    assert all(mapped != original for original, mapped in enumerate(mapping))

    z = torch.randn(1, 2, 3)
    dense = torch.tensor([[0, 1]])
    record = {
        "pair_id": "pair-shuffle",
        "word_id": 2,
        "support_dense_indices": [0],
        "support_weights": [1.0],
        "negative_word_ids": [4, 6],
        "rho": 1.0,
    }
    bank = torch.arange(vocabulary_size * 3, dtype=torch.float32).reshape(
        vocabulary_size, 3
    )
    _, _, positive, negative, valid, _, word_ids = lexical_tensors_from_batch(
        z,
        dense,
        [[record]],
        bank,
        support_mode="shuffled_lexical",
        seed=42,
    )
    expected_positive = _shuffled_lexical_id(2, vocabulary_size, 42)
    expected_negatives = [
        _shuffled_lexical_id(word_id, vocabulary_size, 42) for word_id in (4, 6)
    ]
    assert torch.equal(positive[0], bank[expected_positive])
    assert torch.equal(negative[0, valid[0]], bank[expected_negatives])
    assert word_ids.tolist() == [expected_positive]
