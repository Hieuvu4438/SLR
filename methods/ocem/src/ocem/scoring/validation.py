"""Golden-fixture parity and finite-difference validation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ocem.scoring.autograd import ocem_value
from ocem.scoring.dual_gpu import SolverConfig, centered_solve, centered_solve_bucket
from ocem.scoring.geometry import build_support_geometry
from ocem.scoring.reference import centered_score


INTERVALS = np.array([[0, 2], [1, 3], [8, 10]], dtype=np.float64)
CONCENTRATED = np.array([[0.90, 0.85, -0.20], [0.88, 0.90, -0.20], [0.86, 0.87, -0.20]])
DISTRIBUTED = np.array([[0.90, 0.20, -0.20], [0.20, 0.90, -0.20], [-0.20, 0.10, 0.90]])


def validate_solver(device: str = "cpu", dtype: str = "float64") -> dict[str, Any]:
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    geometry = build_support_geometry(INTERVALS)
    A, w, q = geometry.to_torch(device=device, dtype=torch_dtype)
    config = SolverConfig(
        epsilon=0.05,
        null_prior=0.15,
        kappa=1.5,
        phi_gap_tolerance=1e-8 if torch_dtype == torch.float64 else 1e-4,
        capacity_tolerance=1e-9 if torch_dtype == torch.float64 else 1e-5,
        row_mass_tolerance=1e-8 if torch_dtype == torch.float64 else 1e-6,
        retry_iterations=(128, 512, 2048),
    )
    cases = {}
    started = time.perf_counter()
    for name, array in (("concentrated", CONCENTRATED), ("distributed", DISTRIBUTED)):
        C = torch.tensor(array, dtype=torch_dtype, device=device)
        result = centered_solve(C, A, w, q, config)
        reference = centered_score(array, INTERVALS, epsilon=0.05, null_prior=0.15, kappa=1.5)
        cases[name] = {
            "phi": float(result.phi),
            "reference_phi": float(reference["score"]),
            "absolute_error": abs(float(result.phi) - float(reference["score"])),
            "interval_width": float(result.interval_width),
            "value_gap": float(result.value_result.gap),
            "zero_gap": float(result.zero_result.gap),
            "capacity_violation": float(result.value_result.capacity_violation_before_repair),
            "iterations": result.value_result.iterations,
            "backtracking_steps": result.value_result.backtracking_steps,
            "certified": result.certified,
        }
    C = torch.tensor(CONCENTRATED, dtype=torch_dtype, device=device, requires_grad=True)
    value, result = ocem_value(C, A, w, q, config)
    value.backward()
    step = 1e-5
    finite_difference = np.empty_like(CONCENTRATED)
    for row in range(CONCENTRATED.shape[0]):
        for column in range(CONCENTRATED.shape[1]):
            plus = CONCENTRATED.copy()
            minus = CONCENTRATED.copy()
            plus[row, column] += step
            minus[row, column] -= step
            finite_difference[row, column] = (
                centered_score(
                    plus, INTERVALS, epsilon=0.05, null_prior=0.15, kappa=1.5
                )["score"]
                - centered_score(
                    minus, INTERVALS, epsilon=0.05, null_prior=0.15, kappa=1.5
                )["score"]
            ) / (2 * step)
    envelope_gradient = C.grad.detach().cpu().double().numpy()
    gradient_error = float(np.max(np.abs(finite_difference - envelope_gradient)))
    elapsed = time.perf_counter() - started
    max_value_error = max(case["absolute_error"] for case in cases.values())
    passed = (
        all(case["certified"] for case in cases.values())
        and max_value_error <= (1e-6 if torch_dtype == torch.float64 else 2e-5)
        and gradient_error <= 1e-5
    )
    return {
        "schema_version": "ocem.solver_validation.v1",
        "status": "PASS" if passed else "FAIL_TECHNICAL",
        "device": str(device),
        "dtype": dtype,
        "certificate_profile": "evaluation" if torch_dtype == torch.float64 else "training",
        "cases": cases,
        "max_value_absolute_error": max_value_error,
        "gradient": {
            "entries_checked": int(finite_difference.size),
            "max_absolute_error": gradient_error,
        },
        "elapsed_seconds": elapsed,
        "reference_source_sha256": "5579b07c5fa6e4f4e15563d43919f6d3a7ff0336969be6c86d81812daf687773",
        "kind": "synthetic mathematical validation; NOT an SLRet experiment",
    }


def profile_solver(device: str = "cuda", dtype: str = "float32", pairs: int = 2) -> dict[str, Any]:
    if pairs <= 0:
        raise ValueError("profile pairs must be positive")
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return {
            "status": "BLOCKED_RESOURCE",
            "device": device,
            "dtype": dtype,
            "reason": "CUDA unavailable",
        }
    config = SolverConfig(
        epsilon=0.05,
        null_prior=0.15,
        kappa=1.5,
        phi_gap_tolerance=1e-4 if torch_dtype == torch.float32 else 1e-8,
        capacity_tolerance=1e-5 if torch_dtype == torch.float32 else 1e-9,
        row_mass_tolerance=1e-6 if torch_dtype == torch.float32 else 1e-10,
        retry_iterations=(128, 512, 2048, 4096),
    )
    # Warm up the execution path separately from timed shapes.
    warm_geometry = build_support_geometry([[index, index + 4] for index in range(4)])
    warm_A, warm_w, warm_q = warm_geometry.to_torch(device=device, dtype=torch_dtype)
    warm_C = torch.zeros((1, 2, 4), device=device, dtype=torch_dtype)
    centered_solve(
        warm_C[0], warm_A, warm_w, warm_q, config
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    profiles = []
    for text_tokens, windows in ((8, 16), (16, 32), (32, 64)):
        intervals = [[index, index + 16] for index in range(windows)]
        geometry = build_support_geometry(intervals)
        A, w, q = geometry.to_torch(device=device, dtype=torch_dtype)
        generator = torch.Generator().manual_seed(text_tokens * 100 + windows)
        C = (torch.rand((pairs, text_tokens, windows), generator=generator) * 2 - 1).to(
            device=device, dtype=torch_dtype
        )
        bucket_A = A.expand(pairs, -1, -1).clone()
        bucket_w = w.expand(pairs, -1).clone()
        bucket_q = q.expand(pairs, -1).clone()
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = centered_solve_bucket(C, bucket_A, bucket_w, bucket_q, config)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        profiles.append(
            {
                "text_tokens": text_tokens,
                "windows": windows,
                "atoms": geometry.A.shape[0],
                "pairs": pairs,
                "elapsed_seconds": elapsed,
                "pairs_per_second": pairs / elapsed,
                "certified": result.certified,
                "iterations": result.value_result.iterations.cpu().tolist(),
                "max_iterations": int(result.value_result.iterations.max().cpu()),
                "max_backtracking_steps": int(
                    result.value_result.backtracking_steps.max().cpu()
                ),
                "fallback_count": int(result.value_result.fallback_used.sum().cpu()),
                "max_interval_width": float(result.interval_width.max().cpu()),
                "max_capacity_violation": float(
                    result.value_result.capacity_violation_before_repair.max().cpu()
                ),
                "peak_cuda_memory_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if str(device).startswith("cuda")
                    else None
                ),
            }
        )
    return {
        "status": "PASS" if all(item["certified"] for item in profiles) else "FAIL_TECHNICAL",
        "device": device,
        "dtype": dtype,
        "pairs_per_shape": pairs,
        "retry_iterations": list(config.retry_iterations),
        "profiles": profiles,
        "kind": "synthetic solver profiling; NOT an SLRet experiment",
    }
