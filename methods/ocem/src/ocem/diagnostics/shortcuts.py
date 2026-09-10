"""Offline lexical, duration, token-length, and duplicate controls."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Mapping, Sequence

import numpy as np


_WORD = re.compile(r"\w+", flags=re.UNICODE)


def lexical_tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(str(text).casefold()))


def lexical_jaccard(left: str, right: str) -> float:
    a, b = lexical_tokens(left), lexical_tokens(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def quantile_bin_edges(values: Sequence[float]) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 4 or not np.isfinite(data).all():
        raise ValueError("bin construction needs at least four finite train values")
    return [float(value) for value in np.quantile(data, [0.25, 0.5, 0.75], method="linear")]


def assign_quantile_bin(value: float, edges: Sequence[float]) -> int:
    if len(edges) != 3 or not np.isfinite(value) or not np.isfinite(edges).all():
        raise ValueError("a finite value and three finite bin edges are required")
    return int(np.searchsorted(np.asarray(edges), value, side="right"))


def deterministic_negative_overlaps(
    sample_ids: Sequence[str], captions: Mapping[str, str], *, seed: int
) -> list[float]:
    """Choose one train-only negative per ID without consulting validation outcomes."""

    ids = list(map(str, sample_ids))
    if len(ids) < 2 or len(set(ids)) != len(ids) or set(ids) != set(captions):
        raise ValueError("negative-overlap IDs/captions must be unique and aligned")
    result = []
    for index, sample_id in enumerate(ids):
        digest = hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()
        offset = 1 + int.from_bytes(digest[:8], "big") % (len(ids) - 1)
        other = ids[(index + offset) % len(ids)]
        result.append(lexical_jaccard(captions[sample_id], captions[other]))
    return result


def caption_class_sizes(caption_hashes: Sequence[str]) -> dict[str, int]:
    return dict(Counter(map(str, caption_hashes)))
