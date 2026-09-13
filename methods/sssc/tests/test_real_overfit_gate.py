from __future__ import annotations

import pytest

from tools.run_real_overfit_gate import OverfitGateError, _validate_gate_request


@pytest.mark.parametrize(
    ("steps", "batch_size", "message"),
    [
        (1, 2, "steps must be at least two"),
        (2, 1, "batch size must be at least two"),
        (2, 17, "capped at 16 samples"),
    ],
)
def test_real_overfit_gate_rejects_invalid_request(
    steps: int, batch_size: int, message: str
) -> None:
    with pytest.raises(OverfitGateError, match=message):
        _validate_gate_request(steps=steps, batch_size=batch_size)


@pytest.mark.parametrize(("steps", "batch_size"), [(2, 2), (25, 2), (2, 16)])
def test_real_overfit_gate_accepts_boundary_requests(steps: int, batch_size: int) -> None:
    _validate_gate_request(steps=steps, batch_size=batch_size)
