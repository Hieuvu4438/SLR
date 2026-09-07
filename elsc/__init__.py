"""Source-tree compatibility package for :mod:`methods.elsc.elsc`.

New code lives in ``methods/elsc``. Extending the package search path keeps
the historical ``elsc.*`` CLI and imports working for existing experiment
queues and external run manifests.
"""

from pathlib import Path


_METHOD_PACKAGE = Path(__file__).resolve().parents[1] / "methods" / "elsc" / "elsc"
if not _METHOD_PACKAGE.is_dir():
    raise ImportError(f"ELSC method package is missing: {_METHOD_PACKAGE}")
__path__.append(str(_METHOD_PACKAGE))

__version__ = "0.1.0"
