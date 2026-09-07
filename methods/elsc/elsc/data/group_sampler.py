"""Compatibility alias for shared caption-group sampling."""

import sys

from slr_common.data import group_sampler as _implementation


sys.modules[__name__] = _implementation
