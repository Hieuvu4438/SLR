"""Compatibility alias for historical ``elsc.utils`` imports."""

import sys

from slr_common import utils as _implementation


sys.modules[__name__] = _implementation
