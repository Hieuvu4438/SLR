"""Compatibility alias for shared word/BPE span mapping."""

import sys

from slr_common.data import word_offsets as _implementation


sys.modules[__name__] = _implementation
