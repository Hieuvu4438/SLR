"""Compatibility alias for the shared CiCo feature dataset."""

import sys

from slr_common.data import cico_dataset as _implementation


sys.modules[__name__] = _implementation
