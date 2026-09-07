"""Compatibility CLI for shared release-feature validation."""

import sys

from slr_common.features import validate_release as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
