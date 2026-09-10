"""Exact shared-support geometry over half-open local feature intervals."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


class GeometryError(ValueError):
    """Raised when local temporal supports cannot define valid OCEM geometry."""


@dataclass(frozen=True)
class SupportGeometry:
    intervals: np.ndarray
    atoms: np.ndarray
    A: np.ndarray
    w: np.ndarray
    q: np.ndarray
    geometry_id: str
    version: str = "actual_window_support_v1"

    def to_torch(self, *, device: Any = None, dtype: Any = None):
        import torch

        selected_dtype = dtype if dtype is not None else torch.float64
        return (
            torch.as_tensor(self.A, dtype=selected_dtype, device=device),
            torch.as_tensor(self.w, dtype=selected_dtype, device=device),
            torch.as_tensor(self.q, dtype=selected_dtype, device=device),
        )


def build_support_geometry(intervals: Any) -> SupportGeometry:
    windows = np.asarray(intervals, dtype=np.float64)
    if windows.ndim != 2 or windows.shape[1] != 2 or windows.shape[0] == 0:
        raise GeometryError("intervals must have shape [M, 2] with M > 0")
    if not np.all(np.isfinite(windows)):
        raise GeometryError("interval endpoints must be finite")
    if np.any(windows[:, 1] <= windows[:, 0]):
        raise GeometryError("every interval must have positive width")

    unique, inverse, multiplicities = np.unique(
        windows, axis=0, return_inverse=True, return_counts=True
    )
    endpoints = np.unique(unique.reshape(-1))
    candidate_atoms = np.stack((endpoints[:-1], endpoints[1:]), axis=1)
    centers = candidate_atoms.mean(axis=1)
    membership = (centers[:, None] >= unique[None, :, 0]) & (
        centers[:, None] < unique[None, :, 1]
    )
    covered = membership.any(axis=1)
    atoms = candidate_atoms[covered]
    membership = membership[covered]
    if atoms.shape[0] == 0:
        raise GeometryError("interval union has no positive-width covered atom")
    lengths = atoms[:, 1] - atoms[:, 0]
    union_length = lengths.sum()
    if not np.isfinite(union_length) or union_length <= 0:
        raise GeometryError("invalid interval-union length")
    w = lengths / union_length
    coverage_count = membership.sum(axis=1)
    q_unique = (w[:, None] * membership / coverage_count[:, None]).sum(axis=0)
    q = q_unique[inverse] / multiplicities[inverse]
    widths = windows[:, 1] - windows[:, 0]
    A = lengths[:, None] * membership[:, inverse] / widths[None, :]

    if not np.allclose(A.sum(axis=0), 1.0, atol=1e-12, rtol=0):
        raise GeometryError("geometry incidence columns do not sum to one")
    if not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0):
        raise GeometryError("atom weights do not sum to one")
    if not np.isclose(q.sum(), 1.0, atol=1e-12, rtol=0) or np.any(q <= 0):
        raise GeometryError("reference mass must be positive and sum to one")

    digest = hashlib.sha256()
    digest.update(b"actual_window_support_v1\0")
    digest.update(windows.astype("<f8", copy=False).tobytes(order="C"))
    return SupportGeometry(
        intervals=windows,
        atoms=atoms,
        A=A,
        w=w,
        q=q,
        geometry_id=digest.hexdigest(),
    )
