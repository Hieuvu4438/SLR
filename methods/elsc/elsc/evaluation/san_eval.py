"""Compatibility CLI for the shared SAN protocol adapter."""

import sys

from slr_common.evaluation import san_eval as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
