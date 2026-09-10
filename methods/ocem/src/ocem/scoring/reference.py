"""CPU reference for a proposed SLRet score, not a trained retrieval model.

Dependencies: NumPy and SciPy. Run this file to check mathematical properties.
The score uses pre-contextual visual features whose actual time supports are known.
No dataset, pretrained checkpoint, pose model, or SEDS artifact is needed here.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def support_geometry(intervals):
    """A[a,i] is the fraction of window i in disjoint time atom a.

    Duplicate supports share the reference mass of their canonical support.
    This alone does not imply invariance to changing video frame rate or content.
    """
    windows = np.asarray(intervals, dtype=np.float64)
    if windows.ndim != 2 or windows.shape[1] != 2:
        raise ValueError("intervals must have shape [windows, 2]")
    if not np.all(np.isfinite(windows)) or np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("each interval must have finite, increasing endpoints")
    unique, inverse, count = np.unique(windows, axis=0, return_inverse=True,
                                       return_counts=True)
    ends = np.unique(unique.ravel())
    atoms = np.stack([ends[:-1], ends[1:]], axis=1)
    centers = atoms.mean(axis=1)
    membership = ((centers[:, None] >= unique[None, :, 0]) &
                  (centers[:, None] < unique[None, :, 1]))
    keep = membership.any(axis=1)
    atoms, membership = atoms[keep], membership[keep]
    lengths = atoms[:, 1] - atoms[:, 0]
    weights = lengths / lengths.sum()
    unique_q = (weights[:, None] * membership /
                membership.sum(axis=1)[:, None]).sum(axis=0)
    q = unique_q[inverse] / count[inverse]
    widths = windows[:, 1] - windows[:, 0]
    A = lengths[:, None] * membership[:, inverse] / widths[None, :]
    assert np.allclose(A.sum(axis=0), 1)
    assert np.isclose(q.sum(), 1)
    return A, weights, q


def solve_assignment(similarity, A, weights, q, *, epsilon=0.05,
                     null_prior=0.15, kappa=1.5, maxiter=2000):
    """Maximize <P,C> - eps KL(P || b x [(1-pi)q, pi]).

    Text rows sum to b=1/m, and A @ P_real.sum(0) <= kappa*weights.
    Null similarity is zero. Return a feasible lower bound and a dual upper bound.
    At a sufficiently accurate optimum, d(value)/d(C) = P_real (envelope theorem).
    """
    C = np.asarray(similarity, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != len(q) or C.shape[0] == 0:
        raise ValueError("similarity must have shape [text_tokens, visual_windows]")
    if not np.all(np.isfinite(C)):
        raise ValueError("nonfinite similarities")
    if not 0 < null_prior < 1 or epsilon <= 0 or kappa <= 0:
        raise ValueError("need 0<null_prior<1, epsilon>0, kappa>0")
    b = np.full(C.shape[0], 1.0 / C.shape[0])
    ref = np.r_[(1.0 - null_prior) * q, null_prior]
    logref = np.log(ref)
    cap = kappa * weights

    def evaluate(mu):
        logits = np.concatenate([(C - A.T @ mu) / epsilon,
                                 np.zeros((len(b), 1))], axis=1) + logref
        logz = logsumexp(logits, axis=1)
        P = b[:, None] * np.exp(logits - logz[:, None])
        dual = epsilon * np.dot(b, logz) + np.dot(cap, mu)
        grad = cap - A @ P[:, :-1].sum(axis=0)
        return float(dual), grad, P

    result = minimize(lambda mu: evaluate(mu)[:2], np.zeros(len(weights)),
                      jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(weights),
                      options={"maxiter": maxiter, "ftol": 1e-14,
                               "gtol": 1e-11, "maxls": 50})
    dual, _, P = evaluate(result.x)
    before = A @ P[:, :-1].sum(axis=0)
    # Transfer any tiny infeasible residual to null. This yields a valid lower bound.
    scale = min(1.0, float(np.min(cap / np.maximum(before, 1e-300))))
    removed = (1.0 - scale) * P[:, :-1].sum(axis=1)
    P[:, :-1] *= scale
    P[:, -1] += removed
    R = b[:, None] * ref[None, :]
    kl = np.sum(P * np.log(np.maximum(P, 1e-300) / R))
    primal = float(np.sum(P[:, :-1] * C) - epsilon * kl)
    residual = float(np.maximum(A @ P[:, :-1].sum(0) - cap, 0).max())
    return {"value": primal, "dual_upper": dual, "gap": dual - primal,
            "residual": residual, "iterations": int(result.nit),
            "optimizer_success": bool(result.success), "plan": P}


def centered_score(similarity, intervals, **kwargs):
    A, w, q = support_geometry(intervals)
    answer = solve_assignment(similarity, A, w, q, **kwargs)
    zero = solve_assignment(np.zeros_like(similarity), A, w, q, **kwargs)
    answer["score"] = answer["value"] - zero["value"]
    answer["zero_gap"] = zero["gap"]
    return answer


def mathematical_checks():
    intervals = np.array([[0, 2], [1, 3], [8, 10]], dtype=float)
    concentrated = np.array([[.90, .85, -.20], [.88, .90, -.20], [.86, .87, -.20]])
    distributed = np.array([[.90, .20, -.20], [.20, .90, -.20], [-.20, .10, .90]])
    kw = dict(epsilon=.05, null_prior=.15, kappa=1.5)
    original = centered_score(concentrated, intervals, **kw)
    duplicate = centered_score(np.column_stack([concentrated, concentrated[:, 0]]),
                               np.vstack([intervals, intervals[0]]), **kw)
    loose = centered_score(concentrated, intervals, **{**kw, "kappa": 100})
    spread = centered_score(distributed, intervals, **kw)
    permutation = np.array([2, 0, 1])
    permuted = centered_score(concentrated[permutation], intervals, **kw)
    A, w, q = support_geometry(intervals)
    step = 1e-5
    Cp, Cm = concentrated.copy(), concentrated.copy()
    Cp[1, 0] += step
    Cm[1, 0] -= step
    fd = (solve_assignment(Cp, A, w, q, **kw)["value"] -
          solve_assignment(Cm, A, w, q, **kw)["value"]) / (2 * step)
    expected = original["plan"][1, 0]
    facts = {
        "kind": "synthetic mathematical checks; NOT SLRet experiments",
        "concentrated_score": original["score"],
        "same_affinities_loose_capacity_score": loose["score"],
        "distributed_score": spread["score"],
        "duplicate_invariance_error": abs(original["score"] - duplicate["score"]),
        "text_row_permutation_error": abs(original["score"] - permuted["score"]),
        "capacity_violation": original["residual"],
        "duality_gap": original["gap"],
        "envelope_finite_difference_error": abs(fd - expected),
        "solver_iterations": original["iterations"]
    }
    assert facts["duplicate_invariance_error"] < 1e-7
    assert facts["text_row_permutation_error"] < 1e-7
    assert facts["capacity_violation"] < 1e-9
    assert facts["duality_gap"] < 1e-7
    assert facts["envelope_finite_difference_error"] < 1e-5
    assert original["score"] <= loose["score"] + 1e-7
    return facts


if __name__ == "__main__":
    results = mathematical_checks()
    print(json.dumps(results, indent=2))
    Path(__file__).with_name("ocem_mathematical_checks.json").write_text(
        json.dumps(results, indent=2) + "\n")
