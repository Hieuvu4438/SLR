from __future__ import annotations

import pytest

from elsc.data.cache_dataset import CacheMismatchError, validate_cache_meta


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
