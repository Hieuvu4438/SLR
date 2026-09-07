"""Compatibility alias for shared CiCo metrics."""

import sys

from slr_common.evaluation import cico_eval as _implementation


sys.modules[__name__] = _implementation
