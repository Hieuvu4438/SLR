"""Compatibility alias for shared manifest contracts."""

import sys

from slr_common.data import manifest as _implementation


sys.modules[__name__] = _implementation
