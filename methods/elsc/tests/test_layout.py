from __future__ import annotations

import os
from pathlib import Path

from elsc.config import config_hash, load_config


ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = ROOT / "methods" / "elsc"


def test_method_configs_are_canonical_and_legacy_shims_are_equivalent():
    canonical = sorted((METHOD_ROOT / "configs").glob("*.yaml"))
    assert canonical
    for path in canonical:
        shim = ROOT / "configs" / path.name
        assert shim.is_file()
        assert config_hash(load_config(path, validate=False)) == config_hash(
            load_config(shim, validate=False)
        )


def test_method_campaigns_are_canonical_and_legacy_wrappers_are_executable():
    canonical = sorted((METHOD_ROOT / "scripts").glob("*.sh"))
    assert canonical
    for path in canonical:
        wrapper = ROOT / "scripts" / path.name
        assert wrapper.is_file()
        assert os.access(path, os.X_OK)
        assert os.access(wrapper, os.X_OK)
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert f"methods/elsc/scripts/{path.name}" in wrapper_text


def test_shared_baselines_remain_outside_method_folder():
    for name in ("ph_base.yaml", "ph_release.yaml", "csl_base.yaml"):
        assert (ROOT / "configs" / name).is_file()
        assert not (METHOD_ROOT / "configs" / name).exists()
