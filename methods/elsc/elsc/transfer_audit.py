"""Compatibility CLI for shared transfer-dataset auditing."""

import sys

from slr_common import transfer_audit as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
