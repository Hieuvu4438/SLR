from __future__ import annotations

import pytest
import torch

from dive.evidence_warmup import EvidenceWarmupError, collate_cached_text_units


def test_cached_text_units_collate_variable_lengths_with_explicit_validity():
    cache = {
        "t0": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "t1": torch.tensor([[0.6, 0.8]]),
    }
    values, valid = collate_cached_text_units(("t1", "t0"), cache, device=torch.device("cpu"))
    assert values.shape == (2, 2, 2)
    assert valid.tolist() == [[True, False], [True, True]]
    assert torch.equal(values[0, 1], torch.zeros(2))
    with pytest.raises(EvidenceWarmupError, match="missing IDs"):
        collate_cached_text_units(("missing",), cache, device=torch.device("cpu"))
