from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def pad_feature_sequences(sequences: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad `[length,dim]` sequences with zeros and return `True=valid` masks."""
    if not sequences:
        raise ValueError("cannot collate an empty sequence list")
    if any(item.ndim != 2 for item in sequences):
        raise ValueError("each feature sequence must have shape [length,dim]")
    if any(len(item) == 0 for item in sequences):
        raise ValueError("zero-length sequences must be handled as invalid samples before collate")
    dimensions = {item.shape[1] for item in sequences}
    dtypes = {item.dtype for item in sequences}
    devices = {item.device for item in sequences}
    if len(dimensions) != 1 or len(dtypes) != 1 or len(devices) != 1:
        raise ValueError("all sequences must share feature width, dtype, and device")
    batch = len(sequences)
    maximum = max(len(item) for item in sequences)
    dimension = next(iter(dimensions))
    padded = sequences[0].new_zeros((batch, maximum, dimension))
    valid = torch.zeros((batch, maximum), dtype=torch.bool, device=sequences[0].device)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
        valid[row, : len(sequence)] = True
    return padded, valid
