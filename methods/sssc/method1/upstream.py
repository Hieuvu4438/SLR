from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_upret_tokenizer_module(upret_root: str | Path) -> ModuleType:
    path = Path(upret_root) / "modules" / "tokenization_clip.py"
    if not path.is_file():
        raise FileNotFoundError(f"pinned UPRet tokenizer is missing: {path}")
    spec = importlib.util.spec_from_file_location("method1_pinned_upret_tokenizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned UPRet tokenizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_upret_tokenizer(upret_root: str | Path, bpe_path: str | Path):
    module = load_upret_tokenizer_module(upret_root)
    return module.SimpleTokenizer(str(bpe_path))
