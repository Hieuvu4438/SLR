"""Compatibility namespace for shared data and ELSC cache loading."""

from pathlib import Path


_METHOD_DATA = (
    Path(__file__).resolve().parents[2] / "methods" / "elsc" / "elsc" / "data"
)
if _METHOD_DATA.is_dir():
    __path__.append(str(_METHOD_DATA))
