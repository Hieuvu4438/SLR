"""Source-tree compatibility package for shared sign-retrieval infrastructure."""

from pathlib import Path


_SHARED_PACKAGE = Path(__file__).resolve().parents[1] / "shared" / "slr_common"
if not _SHARED_PACKAGE.is_dir():
    raise ImportError(f"shared package is missing: {_SHARED_PACKAGE}")
__path__.append(str(_SHARED_PACKAGE))
