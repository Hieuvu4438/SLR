"""Compatibility alias for the shared CiCo bridge."""

import sys

from slr_common.upstream import cico_bridge as _implementation


sys.modules[__name__] = _implementation
