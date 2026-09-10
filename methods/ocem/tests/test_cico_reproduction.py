from __future__ import annotations

import pickle

import numpy as np

from ocem.baselines.cico_reproduction import _selected_feature, _tokenize


class _Tokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, words):
        result = []
        for word in words:
            if word == "<|startoftext|>":
                result.append(0)
            elif word == "<|endoftext|>":
                result.append(999)
            else:
                result.append(int(word[1:]) + 1)
        return result


def test_reproduction_token_selection_matches_cico_linspace_rule() -> None:
    ids, valid = _tokenize(_Tokenizer(), " ".join(f"w{i}" for i in range(50)))
    assert len(ids) == len(valid) == 32
    assert all(valid)
    assert ids[0] == 0
    assert ids[-1] == 999
    expected_words = np.linspace(0, 49, 30, dtype=int) + 1
    assert np.array_equal(ids[1:-1], expected_words)


def test_reproduction_feature_selection_covers_short_and_long_cases(tmp_path) -> None:
    short_path = tmp_path / "short.pkl"
    long_path = tmp_path / "long.pkl"
    with short_path.open("wb") as handle:
        pickle.dump({"feature": np.arange(3 * 1024).reshape(3, 1024)}, handle)
    with long_path.open("wb") as handle:
        pickle.dump({"feature": np.arange(100 * 1024).reshape(100, 1024)}, handle)
    short, short_valid = _selected_feature(short_path, feature_len=4)
    long, long_valid = _selected_feature(long_path, feature_len=4)
    assert short.shape == long.shape == (4, 1024)
    assert short_valid.tolist() == [True, True, True, False]
    assert np.all(short[-1] == 0)
    expected = np.linspace(0, 99, 4, dtype=int)
    assert long_valid.tolist() == [True, True, True, True]
    assert np.array_equal(long, np.arange(100 * 1024).reshape(100, 1024)[expected])
