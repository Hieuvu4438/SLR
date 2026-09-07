"""Compatibility alias for shared gallery evaluation runtime."""

import sys

from slr_common.evaluation import runtime as _implementation


sys.modules[__name__] = _implementation
