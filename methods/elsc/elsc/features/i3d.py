"""Compatibility CLI for shared I3D feature extraction."""

import sys

from slr_common.features import i3d as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
