"""Compatibility alias for historical ``elsc.resources`` imports."""

import sys

from slr_common import resources as _implementation


sys.modules[__name__] = _implementation
