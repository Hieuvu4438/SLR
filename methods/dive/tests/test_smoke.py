from __future__ import annotations

import json
from pathlib import Path

from dive.config import load_config
from dive.smoke import run_fixture_smoke


ROOT = Path(__file__).resolve().parents[3]


def test_fixture_smoke_runs_step_checkpoint_reload_and_report(tmp_path):
    config = load_config(ROOT / "methods" / "dive" / "configs" / "fixture.yaml")
    report = run_fixture_smoke(config, tmp_path, repository_root=ROOT)
    assert report["fixture_only"] is True
    assert report["benchmark_claim_allowed"] is False
    assert report["checkpoint_round_trip"] is True
    assert report["resume_global_step"] == 1
    assert report["step_metrics"]["num_sampled"] == 2
    saved = json.loads((tmp_path / "smoke_report.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_sha256"] == report["checkpoint_sha256"]
