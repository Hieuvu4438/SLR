from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared" / "slr_common"
METHOD = ROOT / "methods" / "elsc" / "elsc"


def test_shared_layer_never_imports_a_method_package():
    violations: list[str] = []
    for path in SHARED.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            if roots & {"elsc", "methods"}:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


def test_method_and_shared_sources_have_distinct_canonical_roots():
    assert (METHOD / "train.py").is_file()
    assert (METHOD / "losses" / "evidence.py").is_file()
    assert (SHARED / "features" / "i3d.py").is_file()
    assert (SHARED / "evaluation" / "cico_eval.py").is_file()
