"""OCEM geometry, solvers, controls, and envelope gradients."""

from ocem.scoring.autograd import ocem_value, ocem_values
from ocem.scoring.dual_gpu import (
    BucketCenteredSolveResult,
    CenteredSolveResult,
    SolverConfig,
    centered_solve,
    centered_solve_bucket,
)
from ocem.scoring.geometry import SupportGeometry, build_support_geometry

__all__ = [
    "BucketCenteredSolveResult",
    "CenteredSolveResult",
    "SolverConfig",
    "SupportGeometry",
    "build_support_geometry",
    "centered_solve",
    "centered_solve_bucket",
    "ocem_value",
    "ocem_values",
]
