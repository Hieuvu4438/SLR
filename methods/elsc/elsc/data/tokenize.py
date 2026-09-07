"""Compatibility alias for shared CiCo tokenization."""

import sys

from slr_common.data import tokenize as _implementation


sys.modules[__name__] = _implementation
