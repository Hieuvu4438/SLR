"""Certified projected-gradient solver for the OCEM convex dual.

The implementation supports CPU and CUDA tensors through the same PyTorch
operations. Solver iterations run without autograd; encoder gradients use the
envelope assignment exposed by :mod:`ocem.scoring.autograd`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
import torch


class SolverError(RuntimeError):
    """Raised when inputs are invalid or a certified score cannot be produced."""


@dataclass(frozen=True)
class SolverConfig:
    epsilon: float = 0.05
    null_prior: float = 0.15
    kappa: float = 1.5
    phi_gap_tolerance: float = 1e-4
    capacity_tolerance: float = 1e-5
    row_mass_tolerance: float = 1e-6
    retry_iterations: tuple[int, ...] = (128, 512, 2048, 4096)
    backtracking_limit: int = 64
    certificate_interval: int = 8
    cpu_reference_fallback: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise SolverError("epsilon must be finite and positive")
        if not math.isfinite(self.null_prior) or not 0 < self.null_prior < 1:
            raise SolverError("null_prior must satisfy 0 < pi < 1")
        if not math.isfinite(self.kappa) or self.kappa <= 0:
            raise SolverError("kappa must be finite and positive")
        if (
            self.phi_gap_tolerance <= 0
            or self.capacity_tolerance <= 0
            or self.row_mass_tolerance <= 0
        ):
            raise SolverError("certificate tolerances must be positive")
        if not self.retry_iterations or any(value <= 0 for value in self.retry_iterations):
            raise SolverError("retry_iterations must contain positive budgets")
        if tuple(sorted(self.retry_iterations)) != self.retry_iterations:
            raise SolverError("retry_iterations must be increasing")
        if self.backtracking_limit <= 0 or self.certificate_interval <= 0:
            raise SolverError("backtracking_limit and certificate_interval must be positive")


@dataclass(frozen=True)
class SolveResult:
    primal_value: torch.Tensor
    dual_upper: torch.Tensor
    gap: torch.Tensor
    plan: torch.Tensor
    plan_real: torch.Tensor
    mu: torch.Tensor
    capacity_violation_before_repair: torch.Tensor
    capacity_violation_after_repair: torch.Tensor
    row_mass_error: torch.Tensor
    iterations: int
    converged: bool
    fallback_used: bool
    backtracking_steps: int


@dataclass(frozen=True)
class CenteredSolveResult:
    phi: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor
    interval_width: torch.Tensor
    value_result: SolveResult
    zero_result: SolveResult

    @property
    def certified(self) -> bool:
        return self.value_result.converged and self.zero_result.converged


@dataclass(frozen=True)
class BucketSolveResult:
    primal_value: torch.Tensor
    dual_upper: torch.Tensor
    gap: torch.Tensor
    plan: torch.Tensor
    plan_real: torch.Tensor
    mu: torch.Tensor
    capacity_violation_before_repair: torch.Tensor
    capacity_violation_after_repair: torch.Tensor
    row_mass_error: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    fallback_used: torch.Tensor
    backtracking_steps: torch.Tensor


@dataclass(frozen=True)
class BucketCenteredSolveResult:
    phi: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor
    interval_width: torch.Tensor
    value_result: BucketSolveResult
    zero_result: BucketSolveResult

    @property
    def certified(self) -> bool:
        return bool(self.value_result.converged.all() and self.zero_result.converged.all())


def _validate_inputs(C: torch.Tensor, A: torch.Tensor, w: torch.Tensor, q: torch.Tensor) -> None:
    if C.ndim != 2 or C.shape[0] == 0 or C.shape[1] == 0:
        raise SolverError("C must have shape [text_tokens, windows] with nonzero dimensions")
    if A.ndim != 2 or A.shape[1] != C.shape[1] or A.shape[0] == 0:
        raise SolverError("A must have shape [atoms, windows] matching C")
    if w.shape != (A.shape[0],) or q.shape != (C.shape[1],):
        raise SolverError("w/q shapes do not match A/C")
    if not (C.device == A.device == w.device == q.device):
        raise SolverError("C, A, w, and q must share a device")
    if not (C.dtype == A.dtype == w.dtype == q.dtype):
        raise SolverError("C, A, w, and q must share a dtype")
    if C.dtype not in {torch.float32, torch.float64}:
        raise SolverError("solver supports float32 or float64 only")
    for name, tensor in (("C", C), ("A", A), ("w", w), ("q", q)):
        if not bool(torch.isfinite(tensor).all()):
            raise SolverError(f"{name} contains NaN or infinity")
    if bool((A < 0).any()) or bool((w <= 0).any()) or bool((q <= 0).any()):
        raise SolverError("A must be nonnegative and w/q strictly positive")
    unit = C.new_tensor(1.0)
    tolerance = 1e-6 if C.dtype == torch.float32 else 1e-12
    if not bool(torch.allclose(A.sum(dim=0), unit.expand(C.shape[1]), atol=tolerance, rtol=0)):
        raise SolverError("A columns must sum to one")
    if not bool(torch.isclose(w.sum(), unit, atol=tolerance, rtol=0)):
        raise SolverError("w must sum to one")
    if not bool(torch.isclose(q.sum(), unit, atol=tolerance, rtol=0)):
        raise SolverError("q must sum to one")


def _validate_bucket_inputs(
    C: torch.Tensor, A: torch.Tensor, w: torch.Tensor, q: torch.Tensor
) -> None:
    if C.ndim != 3 or C.shape[0] == 0:
        raise SolverError("bucket C must have shape [pairs, text_tokens, windows]")
    if A.ndim != 3 or A.shape[0] != C.shape[0] or A.shape[2] != C.shape[2]:
        raise SolverError("bucket A must have shape [pairs, atoms, windows]")
    if w.shape != A.shape[:2] or q.shape != (C.shape[0], C.shape[2]):
        raise SolverError("bucket w/q shapes do not match A/C")
    for index in range(C.shape[0]):
        _validate_inputs(C[index], A[index], w[index], q[index])


def _evaluate(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    mu: torch.Tensor,
    config: SolverConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_tokens = C.shape[0]
    b = C.new_full((text_tokens,), 1.0 / text_tokens)
    real_log_reference = torch.log(q * (1.0 - config.null_prior))
    penalty = A.transpose(0, 1).matmul(mu)
    real_logits = (C - penalty[None, :]) / config.epsilon + real_log_reference[None, :]
    null_logits = C.new_full((text_tokens, 1), math.log(config.null_prior))
    logits = torch.cat((real_logits, null_logits), dim=1)
    logz = torch.logsumexp(logits, dim=1)
    plan = b[:, None] * torch.exp(logits - logz[:, None])
    cap = config.kappa * w
    dual = config.epsilon * torch.dot(b, logz) + torch.dot(cap, mu)
    load = A.matmul(plan[:, :-1].sum(dim=0))
    gradient = cap - load
    return dual, gradient, plan


def _evaluate_bucket(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    mu: torch.Tensor,
    config: SolverConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_tokens = C.shape[1]
    real_log_reference = torch.log(q * (1.0 - config.null_prior))
    penalty = torch.einsum("qlm,ql->qm", A, mu)
    real_logits = (C - penalty[:, None, :]) / config.epsilon + real_log_reference[:, None, :]
    null_logits = C.new_full((C.shape[0], text_tokens, 1), math.log(config.null_prior))
    logits = torch.cat((real_logits, null_logits), dim=2)
    logz = torch.logsumexp(logits, dim=2)
    plan = torch.exp(logits - logz[:, :, None]) / text_tokens
    cap = config.kappa * w
    dual = config.epsilon * logz.mean(dim=1) + torch.sum(cap * mu, dim=1)
    load = torch.einsum("qlm,qm->ql", A, plan[:, :, :-1].sum(dim=1))
    gradient = cap - load
    return dual, gradient, plan


def _primal_certificate(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    plan: torch.Tensor,
    config: SolverConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cap = config.kappa * w
    before_load = A.matmul(plan[:, :-1].sum(dim=0))
    violation_before = torch.clamp_min(before_load - cap, 0).max()
    tiny = torch.finfo(C.dtype).tiny
    scale = torch.minimum(C.new_tensor(1.0), torch.min(cap / torch.clamp_min(before_load, tiny)))
    feasible = plan.clone()
    removed = (1.0 - scale) * feasible[:, :-1].sum(dim=1)
    feasible[:, :-1] = feasible[:, :-1] * scale
    feasible[:, -1] = feasible[:, -1] + removed
    after_load = A.matmul(feasible[:, :-1].sum(dim=0))
    violation_after = torch.clamp_min(after_load - cap, 0).max()
    b = C.new_full((C.shape[0],), 1.0 / C.shape[0])
    reference = torch.cat((q * (1.0 - config.null_prior), q.new_tensor([config.null_prior])))
    R = b[:, None] * reference[None, :]
    safe = torch.clamp_min(feasible, tiny)
    kl = torch.sum(feasible * (torch.log(safe) - torch.log(R)))
    primal = torch.sum(feasible[:, :-1] * C) - config.epsilon * kl
    row_error = torch.max(torch.abs(feasible.sum(dim=1) - b))
    return primal, feasible, violation_before, violation_after, row_error


def _primal_certificate_bucket(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    plan: torch.Tensor,
    config: SolverConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cap = config.kappa * w
    before_load = torch.einsum("qlm,qm->ql", A, plan[:, :, :-1].sum(dim=1))
    violation_before = torch.clamp_min(before_load - cap, 0).max(dim=1).values
    tiny = torch.finfo(C.dtype).tiny
    scale = torch.minimum(
        C.new_ones(C.shape[0]), torch.min(cap / torch.clamp_min(before_load, tiny), dim=1).values
    )
    feasible = plan.clone()
    removed = (1.0 - scale[:, None]) * feasible[:, :, :-1].sum(dim=2)
    feasible[:, :, :-1] = feasible[:, :, :-1] * scale[:, None, None]
    feasible[:, :, -1] = feasible[:, :, -1] + removed
    after_load = torch.einsum("qlm,qm->ql", A, feasible[:, :, :-1].sum(dim=1))
    violation_after = torch.clamp_min(after_load - cap, 0).max(dim=1).values
    reference = torch.cat(
        (q * (1.0 - config.null_prior), q.new_full((q.shape[0], 1), config.null_prior)), dim=1
    )
    R = reference[:, None, :] / C.shape[1]
    safe = torch.clamp_min(feasible, tiny)
    kl = torch.sum(feasible * (torch.log(safe) - torch.log(R)), dim=(1, 2))
    primal = torch.sum(feasible[:, :, :-1] * C, dim=(1, 2)) - config.epsilon * kl
    row_error = torch.max(
        torch.abs(feasible.sum(dim=2) - 1.0 / C.shape[1]), dim=1
    ).values
    return primal, feasible, violation_before, violation_after, row_error


def _result_from_iterate(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    mu: torch.Tensor,
    config: SolverConfig,
    iterations: int,
    backtracking_steps: int,
) -> SolveResult:
    dual, _, plan = _evaluate(C, A, w, q, mu, config)
    primal, feasible, violation_before, violation_after, row_error = _primal_certificate(
        C, A, w, q, plan, config
    )
    gap = dual - primal
    rounding = 64 * torch.finfo(C.dtype).eps * (1.0 + torch.abs(dual))
    if bool(gap < -rounding):
        raise SolverError(f"negative primal-dual gap: {float(gap)}")
    # Preserve interval semantics when round-off puts the dual a few ulps below
    # the independently evaluated feasible primal objective.
    dual = torch.maximum(dual, primal)
    gap = dual - primal
    converged = bool(
        gap <= config.phi_gap_tolerance / 2
        and violation_before <= config.capacity_tolerance
        and violation_after <= config.capacity_tolerance
        and row_error <= config.row_mass_tolerance
    )
    return SolveResult(
        primal_value=primal,
        dual_upper=dual,
        gap=gap,
        plan=feasible,
        plan_real=feasible[:, :-1],
        mu=mu,
        capacity_violation_before_repair=violation_before,
        capacity_violation_after_repair=violation_after,
        row_mass_error=row_error,
        iterations=iterations,
        converged=converged,
        fallback_used=False,
        backtracking_steps=backtracking_steps,
    )


def _reference_fallback(
    C: torch.Tensor, A: torch.Tensor, w: torch.Tensor, q: torch.Tensor, config: SolverConfig
) -> SolveResult:
    from ocem.scoring.reference import solve_assignment

    answer = solve_assignment(
        C.detach().cpu().double().numpy(),
        A.detach().cpu().double().numpy(),
        w.detach().cpu().double().numpy(),
        q.detach().cpu().double().numpy(),
        epsilon=config.epsilon,
        null_prior=config.null_prior,
        kappa=config.kappa,
        maxiter=max(config.retry_iterations),
    )
    plan = torch.as_tensor(answer["plan"], dtype=C.dtype, device=C.device)
    mu = torch.zeros(A.shape[0], dtype=C.dtype, device=C.device)
    result = SolveResult(
        primal_value=C.new_tensor(answer["value"]),
        dual_upper=C.new_tensor(answer["dual_upper"]),
        gap=C.new_tensor(answer["gap"]),
        plan=plan,
        plan_real=plan[:, :-1],
        mu=mu,
        capacity_violation_before_repair=C.new_tensor(answer["residual"]),
        capacity_violation_after_repair=C.new_tensor(answer["residual"]),
        row_mass_error=torch.max(
            torch.abs(plan.sum(dim=1) - C.new_full((C.shape[0],), 1.0 / C.shape[0]))
        ),
        iterations=int(answer["iterations"]),
        converged=bool(
            answer["gap"] <= config.phi_gap_tolerance / 2
            and answer["residual"] <= config.capacity_tolerance
        ),
        fallback_used=True,
        backtracking_steps=0,
    )
    return result


@torch.no_grad()
def _solve_assignment_no_autocast(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> SolveResult:
    config.validate()
    _validate_inputs(C, A, w, q)
    mu = torch.zeros(A.shape[0], dtype=C.dtype, device=C.device)
    squared_spectral_norm = torch.linalg.matrix_norm(A, ord=2) ** 2
    eta = 1000.0 * config.epsilon / (squared_spectral_norm + torch.finfo(C.dtype).eps)
    momentum = mu.clone()
    acceleration = 1.0
    accepted_steps = 0
    backtracking_steps = 0
    last_result = _result_from_iterate(C, A, w, q, mu, config, 0, 0)
    if last_result.converged:
        return last_result
    maximum = max(config.retry_iterations)
    checkpoints = set(config.retry_iterations)
    for iteration in range(1, maximum + 1):
        dual, gradient, _ = _evaluate(C, A, w, q, momentum, config)
        accepted = False
        trial_eta = eta
        for _ in range(config.backtracking_limit):
            candidate = torch.clamp_min(momentum - trial_eta * gradient, 0)
            delta = candidate - momentum
            candidate_dual, _, _ = _evaluate(C, A, w, q, candidate, config)
            rhs = dual + torch.dot(gradient, delta) + torch.dot(delta, delta) / (2 * trial_eta)
            slack = 64 * torch.finfo(C.dtype).eps * (1.0 + torch.abs(dual))
            if bool(candidate_dual <= rhs + slack):
                eta = trial_eta
                accepted = True
                accepted_steps += 1
                break
            trial_eta = trial_eta * 0.5
            backtracking_steps += 1
        if not accepted:
            raise SolverError("projected-gradient backtracking exhausted")
        previous = mu
        mu = candidate
        next_acceleration = (1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration)) / 2.0
        beta = (acceleration - 1.0) / next_acceleration
        proposed_momentum = mu + beta * (mu - previous)
        # Adaptive restart prevents acceleration from oscillating across an active boundary.
        if bool(torch.dot(momentum - mu, mu - previous) > 0):
            momentum = mu.clone()
            acceleration = 1.0
        else:
            momentum = torch.clamp_min(proposed_momentum, 0)
            acceleration = next_acceleration
        should_check = (
            iteration % config.certificate_interval == 0 or iteration in checkpoints or iteration == maximum
        )
        if should_check:
            last_result = _result_from_iterate(
                C, A, w, q, mu, config, iteration, backtracking_steps
            )
            if last_result.converged:
                return last_result
    if config.cpu_reference_fallback:
        return _reference_fallback(C, A, w, q, config)
    return replace(last_result, iterations=accepted_steps)


@torch.no_grad()
def solve_assignment(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> SolveResult:
    with torch.autocast(device_type=C.device.type, enabled=False):
        return _solve_assignment_no_autocast(C, A, w, q, config)


@torch.no_grad()
def centered_solve(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> CenteredSolveResult:
    value = solve_assignment(C, A, w, q, config)
    zero = solve_assignment(torch.zeros_like(C), A, w, q, config)
    phi = value.primal_value - zero.primal_value
    lower = value.primal_value - zero.dual_upper
    upper = value.dual_upper - zero.primal_value
    width = upper - lower
    return CenteredSolveResult(
        phi=phi,
        lower_bound=lower,
        upper_bound=upper,
        interval_width=width,
        value_result=value,
        zero_result=zero,
    )


@torch.no_grad()
def _solve_assignment_bucket_no_autocast(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> BucketSolveResult:
    """Solve an exact-shape bucket with independent per-pair convergence state."""

    config.validate()
    _validate_bucket_inputs(C, A, w, q)
    pairs, atoms = A.shape[:2]
    mu = torch.zeros((pairs, atoms), dtype=C.dtype, device=C.device)
    momentum = mu.clone()
    acceleration = C.new_ones(pairs)
    squared_spectral_norm = torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1)) ** 2
    eta = 1000.0 * config.epsilon / (squared_spectral_norm + torch.finfo(C.dtype).eps)
    initial_dual, _, initial_plan = _evaluate_bucket(C, A, w, q, mu, config)
    initial_primal, _, initial_before, initial_after, initial_row_error = (
        _primal_certificate_bucket(C, A, w, q, initial_plan, config)
    )
    initial_gap = initial_dual - initial_primal
    converged = (
        (initial_gap <= config.phi_gap_tolerance / 2)
        & (initial_before <= config.capacity_tolerance)
        & (initial_after <= config.capacity_tolerance)
        & (initial_row_error <= config.row_mass_tolerance)
    )
    iterations = torch.zeros(pairs, dtype=torch.int64, device=C.device)
    backtracking = torch.zeros(pairs, dtype=torch.int64, device=C.device)
    maximum = 0 if bool(converged.all()) else max(config.retry_iterations)
    checkpoints = set(config.retry_iterations)
    for iteration in range(1, maximum + 1):
        active = ~converged
        dual, gradient, _ = _evaluate_bucket(C, A, w, q, momentum, config)
        accepted = ~active
        candidate = mu.clone()
        trial_eta = eta.clone()
        for _ in range(config.backtracking_limit):
            proposed = torch.clamp_min(momentum - trial_eta[:, None] * gradient, 0)
            delta = proposed - momentum
            proposed_dual, _, _ = _evaluate_bucket(C, A, w, q, proposed, config)
            rhs = dual + torch.sum(gradient * delta, dim=1) + torch.sum(delta * delta, dim=1) / (
                2 * trial_eta
            )
            slack = 64 * torch.finfo(C.dtype).eps * (1.0 + torch.abs(dual))
            newly = active & ~accepted & (proposed_dual <= rhs + slack)
            candidate = torch.where(newly[:, None], proposed, candidate)
            accepted = accepted | newly
            rejected = active & ~accepted
            if not bool(rejected.any()):
                break
            trial_eta = torch.where(rejected, trial_eta * 0.5, trial_eta)
            backtracking = backtracking + rejected.to(torch.int64)
        if not bool(accepted.all()):
            failed = torch.nonzero(~accepted, as_tuple=False).flatten().tolist()
            raise SolverError(f"bucket backtracking exhausted for pair indices {failed}")
        previous = mu
        mu = torch.where(active[:, None], candidate, mu)
        eta = torch.where(active, trial_eta, eta)
        next_acceleration = (1.0 + torch.sqrt(1.0 + 4.0 * acceleration * acceleration)) / 2.0
        beta = (acceleration - 1.0) / next_acceleration
        proposed_momentum = mu + beta[:, None] * (mu - previous)
        restart = torch.sum((momentum - mu) * (mu - previous), dim=1) > 0
        momentum = torch.where(
            active[:, None],
            torch.where(restart[:, None], mu, torch.clamp_min(proposed_momentum, 0)),
            mu,
        )
        acceleration = torch.where(
            active, torch.where(restart, torch.ones_like(acceleration), next_acceleration), acceleration
        )
        should_check = (
            iteration % config.certificate_interval == 0 or iteration in checkpoints or iteration == maximum
        )
        if should_check:
            dual_now, _, plan_now = _evaluate_bucket(C, A, w, q, mu, config)
            primal, _, before, after, row_error = _primal_certificate_bucket(
                C, A, w, q, plan_now, config
            )
            gap = dual_now - primal
            rounding = 64 * torch.finfo(C.dtype).eps * (1.0 + torch.abs(dual_now))
            if bool((gap < -rounding).any()):
                raise SolverError("bucket contains a negative primal-dual gap")
            dual_now = torch.maximum(dual_now, primal)
            gap = dual_now - primal
            now = (
                (gap <= config.phi_gap_tolerance / 2)
                & (before <= config.capacity_tolerance)
                & (after <= config.capacity_tolerance)
                & (row_error <= config.row_mass_tolerance)
            )
            newly_converged = active & now
            iterations = torch.where(
                newly_converged, torch.full_like(iterations, iteration), iterations
            )
            converged = converged | now
            if bool(converged.all()):
                break
    dual, _, plan = _evaluate_bucket(C, A, w, q, mu, config)
    primal, feasible, before, after, row_error = _primal_certificate_bucket(
        C, A, w, q, plan, config
    )
    gap = dual - primal
    rounding = 64 * torch.finfo(C.dtype).eps * (1.0 + torch.abs(dual))
    if bool((gap < -rounding).any()):
        raise SolverError("bucket contains a negative final primal-dual gap")
    dual = torch.maximum(dual, primal)
    gap = dual - primal
    iterations = torch.where(
        converged, iterations, torch.full_like(iterations, maximum)
    )
    fallback_used = torch.zeros(pairs, dtype=torch.bool, device=C.device)
    if config.cpu_reference_fallback and bool((~converged).any()):
        for index in torch.nonzero(~converged, as_tuple=False).flatten().tolist():
            fallback = _reference_fallback(C[index], A[index], w[index], q[index], config)
            primal[index] = fallback.primal_value
            dual[index] = fallback.dual_upper
            gap[index] = fallback.gap
            feasible[index] = fallback.plan
            mu[index] = fallback.mu
            before[index] = fallback.capacity_violation_before_repair
            after[index] = fallback.capacity_violation_after_repair
            row_error[index] = fallback.row_mass_error
            iterations[index] = fallback.iterations
            converged[index] = fallback.converged
            fallback_used[index] = True
    return BucketSolveResult(
        primal_value=primal,
        dual_upper=dual,
        gap=gap,
        plan=feasible,
        plan_real=feasible[:, :, :-1],
        mu=mu,
        capacity_violation_before_repair=before,
        capacity_violation_after_repair=after,
        row_mass_error=row_error,
        iterations=iterations,
        converged=converged,
        fallback_used=fallback_used,
        backtracking_steps=backtracking,
    )


@torch.no_grad()
def solve_assignment_bucket(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> BucketSolveResult:
    with torch.autocast(device_type=C.device.type, enabled=False):
        return _solve_assignment_bucket_no_autocast(C, A, w, q, config)


@torch.no_grad()
def centered_solve_bucket(
    C: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    config: SolverConfig = SolverConfig(),
) -> BucketCenteredSolveResult:
    value = solve_assignment_bucket(C, A, w, q, config)
    zero = solve_assignment_bucket(torch.zeros_like(C), A, w, q, config)
    phi = value.primal_value - zero.primal_value
    lower = value.primal_value - zero.dual_upper
    upper = value.dual_upper - zero.primal_value
    return BucketCenteredSolveResult(
        phi=phi,
        lower_bound=lower,
        upper_bound=upper,
        interval_width=upper - lower,
        value_result=value,
        zero_result=zero,
    )
