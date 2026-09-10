"""First-order envelope backward for certified OCEM values."""

from __future__ import annotations

import torch

from ocem.scoring.dual_gpu import (
    BucketCenteredSolveResult,
    CenteredSolveResult,
    SolverConfig,
    SolverError,
    centered_solve,
    centered_solve_bucket,
)


class _EnvelopeValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C: torch.Tensor, plan_real: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(plan_real)
        return phi

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (plan_real,) = ctx.saved_tensors
        expanded = grad_output
        while expanded.ndim < plan_real.ndim:
            expanded = expanded.unsqueeze(-1)
        return expanded * plan_real, None, None


def ocem_value(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> tuple[torch.Tensor, CenteredSolveResult]:
    """Return a differentiable centered value and detached solver diagnostics."""

    solver_dtype = torch.float64 if C.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=C.device.type, enabled=False):
        result = centered_solve(
            C.detach().to(dtype=solver_dtype),
            A.detach().to(device=C.device, dtype=solver_dtype),
            w.detach().to(device=C.device, dtype=solver_dtype),
            q.detach().to(device=C.device, dtype=solver_dtype),
            config,
        )
    if not result.certified:
        raise SolverError(
            "uncertified OCEM value: "
            f"width={float(result.interval_width)}, "
            f"value_converged={result.value_result.converged}, "
            f"zero_converged={result.zero_result.converged}"
        )
    value = _EnvelopeValue.apply(C, result.value_result.plan_real, result.phi)
    return value, result


def ocem_values(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> tuple[torch.Tensor, BucketCenteredSolveResult]:
    """Batched envelope values for one exact `(m, M, L)` pair bucket."""

    solver_dtype = torch.float64 if C.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=C.device.type, enabled=False):
        result = centered_solve_bucket(
            C.detach().to(dtype=solver_dtype),
            A.detach().to(device=C.device, dtype=solver_dtype),
            w.detach().to(device=C.device, dtype=solver_dtype),
            q.detach().to(device=C.device, dtype=solver_dtype),
            config,
        )
    if not result.certified:
        failed_value = torch.nonzero(~result.value_result.converged).flatten().tolist()
        failed_zero = torch.nonzero(~result.zero_result.converged).flatten().tolist()
        raise SolverError(
            f"uncertified OCEM bucket; value pairs={failed_value}, zero pairs={failed_zero}"
        )
    values = _EnvelopeValue.apply(C, result.value_result.plan_real, result.phi)
    return values, result
