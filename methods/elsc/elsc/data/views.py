"""Compatibility alias for shared temporal feature views."""

import sys

from slr_common.data import views as _implementation


sys.modules[__name__] = _implementation
