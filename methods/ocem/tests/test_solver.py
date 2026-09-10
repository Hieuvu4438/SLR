from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from ocem.scoring.autograd import ocem_value, ocem_values
from ocem.scoring.dual_gpu import (
    SolverConfig,
    SolverError,
    centered_solve,
    centered_solve_bucket,
    solve_assignment,
)
from ocem.scoring.geometry import build_support_geometry
from ocem.scoring.reference import centered_score
from ocem.scoring.validation import CONCENTRATED, DISTRIBUTED, INTERVALS


def _fixture(dtype=torch.float64, device="cpu", **changes):
    geometry = build_support_geometry(INTERVALS)
    A, w, q = geometry.to_torch(dtype=dtype, device=device)
    values = {
        "epsilon": 0.05,
        "null_prior": 0.15,
        "kappa": 1.5,
        "phi_gap_tolerance": 1e-7,
        "capacity_tolerance": 1e-8,
        "row_mass_tolerance": 1e-10,
        "retry_iterations": (128, 512, 2048),
    }
    values.update(changes)
    return A, w, q, SolverConfig(**values)


@pytest.mark.parametrize("array", [CONCENTRATED, DISTRIBUTED])
def test_projected_solver_matches_scipy_reference(array) -> None:
    A, w, q, config = _fixture()
    C = torch.tensor(array, dtype=torch.float64)
    actual = centered_solve(C, A, w, q, config)
    expected = centered_score(array, INTERVALS, epsilon=0.05, null_prior=0.15, kappa=1.5)
    assert actual.certified
    assert float(actual.phi) == pytest.approx(expected["score"], abs=1e-7)
    assert float(actual.interval_width) <= 1e-7
    assert float(actual.value_result.capacity_violation_before_repair) <= 1e-8


def test_unconstrained_inactive_solution_stops_at_zero_dual() -> None:
    A, w, q, config = _fixture(kappa=100.0)
    C = torch.tensor(CONCENTRATED, dtype=torch.float64)
    result = solve_assignment(C, A, w, q, config)
    assert result.converged
    assert result.iterations == 0
    assert torch.count_nonzero(result.mu) == 0


def test_duplicate_feature_support_invariance() -> None:
    A, w, q, config = _fixture()
    C = torch.tensor(CONCENTRATED, dtype=torch.float64)
    original = centered_solve(C, A, w, q, config)
    duplicated_intervals = np.vstack((INTERVALS, INTERVALS[0]))
    duplicated_geometry = build_support_geometry(duplicated_intervals)
    duplicated_A, duplicated_w, duplicated_q = duplicated_geometry.to_torch(dtype=torch.float64)
    duplicated_C = torch.column_stack((C, C[:, 0]))
    duplicated = centered_solve(duplicated_C, duplicated_A, duplicated_w, duplicated_q, config)
    assert float(duplicated.phi) == pytest.approx(float(original.phi), abs=1e-7)


def test_envelope_backward_matches_finite_difference() -> None:
    A, w, q, config = _fixture()
    C = torch.tensor(CONCENTRATED, dtype=torch.float64, requires_grad=True)
    value, result = ocem_value(C, A, w, q, config)
    value.backward()
    assert torch.count_nonzero(C.grad) > 0
    step = 1e-5
    finite_difference = torch.empty_like(C)
    for row in range(C.shape[0]):
        for column in range(C.shape[1]):
            plus, minus = C.detach().clone(), C.detach().clone()
            plus[row, column] += step
            minus[row, column] -= step
            finite_difference[row, column] = (
                centered_solve(plus, A, w, q, config).phi
                - centered_solve(minus, A, w, q, config).phi
            ) / (2 * step)
    torch.testing.assert_close(C.grad, finite_difference, atol=1e-5, rtol=0)
    assert torch.allclose(C.grad, result.value_result.plan_real)


def test_tiny_independent_primal_optimizer_is_bracketed_by_dual() -> None:
    intervals = np.array([[0, 2], [1, 3]], dtype=float)
    geometry = build_support_geometry(intervals)
    C_np = np.array([[0.8, 0.7], [0.6, -0.2]], dtype=float)
    epsilon, null_prior, kappa = 0.1, 0.2, 1.2
    m, windows = C_np.shape
    b = np.full(m, 1.0 / m)
    reference = np.r_[(1 - null_prior) * geometry.q, null_prior]
    R = b[:, None] * reference[None, :]

    def objective(flat):
        P = flat.reshape(m, windows + 1)
        return -(np.sum(P[:, :-1] * C_np) - epsilon * np.sum(P * np.log(P / R)))

    constraints = [
        {"type": "eq", "fun": lambda flat, row=row: flat.reshape(m, windows + 1)[row].sum() - b[row]}
        for row in range(m)
    ]
    for atom in range(geometry.A.shape[0]):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda flat, atom=atom: kappa * geometry.w[atom]
                - geometry.A[atom].dot(flat.reshape(m, windows + 1)[:, :-1].sum(axis=0)),
            }
        )
    initial = R.reshape(-1)
    answer = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * initial.size,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert answer.success
    A, w, q = geometry.to_torch(dtype=torch.float64)
    config = SolverConfig(
        epsilon=epsilon,
        null_prior=null_prior,
        kappa=kappa,
        phi_gap_tolerance=1e-8,
        capacity_tolerance=1e-9,
        row_mass_tolerance=1e-10,
    )
    dual = solve_assignment(torch.tensor(C_np, dtype=torch.float64), A, w, q, config)
    independent_primal = -answer.fun
    assert float(dual.primal_value) <= independent_primal + 1e-7
    assert independent_primal <= float(dual.dual_upper) + 1e-7


def test_exact_shape_bucket_matches_individual_solves_and_gradients() -> None:
    A, w, q, config = _fixture()
    other_geometry = build_support_geometry([[0, 3], [2, 4], [8, 10]])
    other_A, other_w, other_q = other_geometry.to_torch(dtype=torch.float64)
    C = torch.tensor(np.stack((CONCENTRATED, DISTRIBUTED)), dtype=torch.float64, requires_grad=True)
    bucket_A = torch.stack((A, other_A))
    bucket_w = torch.stack((w, other_w))
    bucket_q = torch.stack((q, other_q))
    values, bucket = ocem_values(C, bucket_A, bucket_w, bucket_q, config)
    values.sum().backward()
    individuals = [
        centered_solve(C.detach()[0], A, w, q, config),
        centered_solve(C.detach()[1], other_A, other_w, other_q, config),
    ]
    assert bucket.certified
    torch.testing.assert_close(
        bucket.phi, torch.stack([result.phi for result in individuals]), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        C.grad,
        torch.stack([result.value_result.plan_real for result in individuals]),
        atol=1e-7,
        rtol=0,
    )
    direct = centered_solve_bucket(C.detach(), bucket_A, bucket_w, bucket_q, config)
    torch.testing.assert_close(direct.phi, bucket.phi, atol=1e-12, rtol=0)


def test_low_kappa_unequal_supports_produce_null_heavy_feasible_plan() -> None:
    geometry = build_support_geometry([[0, 4], [1, 2], [3, 10]])
    A, w, q = geometry.to_torch(dtype=torch.float64)
    config = SolverConfig(
        kappa=0.3,
        null_prior=0.15,
        phi_gap_tolerance=1e-8,
        capacity_tolerance=1e-9,
        row_mass_tolerance=1e-10,
    )
    C = torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=torch.float64)
    result = solve_assignment(C, A, w, q, config)
    assert result.converged
    assert not result.fallback_used
    assert float(result.plan_real.sum()) <= 0.3 + 1e-9
    assert float(result.plan[:, -1].sum()) >= 0.7 - 1e-9
    assert float(result.capacity_violation_after_repair) <= 1e-9


def test_single_window_and_zero_center_are_well_defined() -> None:
    geometry = build_support_geometry([[4, 20]])
    A, w, q = geometry.to_torch(dtype=torch.float64)
    config = SolverConfig(
        kappa=1.5,
        phi_gap_tolerance=1e-8,
        capacity_tolerance=1e-9,
        row_mass_tolerance=1e-10,
    )
    for text_tokens in (1, 2, 7):
        zero = centered_solve(torch.zeros((text_tokens, 1), dtype=torch.float64), A, w, q, config)
        assert zero.certified
        assert float(zero.phi) == pytest.approx(0.0, abs=1e-12)
        assert float(zero.interval_width) <= 1e-8


@pytest.mark.parametrize(
    "change",
    [
        {"row_mass_tolerance": 0.0},
        {"backtracking_limit": 0},
        {"certificate_interval": 0},
    ],
)
def test_invalid_solver_config_fails_before_iteration(change) -> None:
    A, w, q, _ = _fixture()
    with pytest.raises(SolverError, match="positive"):
        solve_assignment(
            torch.tensor(CONCENTRATED, dtype=torch.float64), A, w, q, SolverConfig(**change)
        )


def test_window_and_text_permutations_preserve_centered_score() -> None:
    A, w, q, config = _fixture()
    C = torch.tensor(CONCENTRATED, dtype=torch.float64)
    original = centered_solve(C, A, w, q, config)
    window_permutation = torch.tensor([2, 0, 1])
    text_permutation = torch.tensor([2, 0, 1])
    permuted = centered_solve(
        C[text_permutation][:, window_permutation],
        A[:, window_permutation],
        w,
        q[window_permutation],
        config,
    )
    assert float(permuted.phi) == pytest.approx(float(original.phi), abs=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_solver_disables_amp_and_uses_float32_for_half_encoder_affinity() -> None:
    geometry = build_support_geometry(INTERVALS)
    A, w, q = geometry.to_torch(device="cuda", dtype=torch.float64)
    config = SolverConfig(
        phi_gap_tolerance=1e-4,
        capacity_tolerance=1e-5,
        row_mass_tolerance=1e-6,
    )
    C = torch.tensor(CONCENTRATED, dtype=torch.float16, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        value, result = ocem_value(C, A, w, q, config)
    value.backward()
    assert result.phi.dtype == torch.float32
    assert result.value_result.plan.dtype == torch.float32
    assert C.grad is not None and C.grad.dtype == torch.float16
    assert result.certified


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cpu_value_plan_and_gradient_parity() -> None:
    cpu_A, cpu_w, cpu_q, cpu_config = _fixture(dtype=torch.float64, device="cpu")
    gpu_A, gpu_w, gpu_q, gpu_config = _fixture(dtype=torch.float64, device="cuda")
    cpu_C = torch.tensor(CONCENTRATED, dtype=torch.float64, requires_grad=True)
    gpu_C = torch.tensor(CONCENTRATED, dtype=torch.float64, device="cuda", requires_grad=True)
    cpu_value, cpu_result = ocem_value(cpu_C, cpu_A, cpu_w, cpu_q, cpu_config)
    gpu_value, gpu_result = ocem_value(gpu_C, gpu_A, gpu_w, gpu_q, gpu_config)
    cpu_value.backward()
    gpu_value.backward()
    assert float(gpu_value.detach().cpu()) == pytest.approx(float(cpu_value.detach()), abs=1e-7)
    assert torch.allclose(
        gpu_result.value_result.plan_real.cpu(), cpu_result.value_result.plan_real, atol=1e-7, rtol=0
    )
    assert torch.allclose(gpu_C.grad.cpu(), cpu_C.grad, atol=1e-7, rtol=0)
