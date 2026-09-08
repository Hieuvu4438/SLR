from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
CICO_ROOT = ROOT / "third_party" / "SLRT" / "CiCo" / "CLCL"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods" / "dive" / "src"))


@pytest.fixture(scope="session")
def cico_tokenizer():
    sys.path.insert(0, str(CICO_ROOT))
    from modules.tokenization_clip import SimpleTokenizer

    return SimpleTokenizer(str(CICO_ROOT / "modules" / "bpe_simple_vocab_16e6.txt.gz"))


@pytest.fixture(scope="session")
def cico_cleaners():
    sys.path.insert(0, str(CICO_ROOT))
    from modules.tokenization_clip import basic_clean, whitespace_clean

    return basic_clean, whitespace_clean
