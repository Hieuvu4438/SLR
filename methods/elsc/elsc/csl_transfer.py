"""Compatibility CLI for the shared CSL-Daily transfer utility."""

import sys

from slr_common import csl_transfer as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
