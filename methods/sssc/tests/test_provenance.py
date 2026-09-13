from __future__ import annotations

from method1.provenance import runtime_environment_report
from method1.utils import sha256_json


def test_runtime_provenance_captures_lock_freeze_and_determinism() -> None:
    report = runtime_environment_report()
    assert report["status"] == "complete"
    assert len(report["packages"]["pip_freeze_sha256"]) == 64
    assert report["packages"]["pip_freeze"]
    assert len(report["target_lock"]["sha256"]) == 64
    assert isinstance(report["determinism"]["deterministic_algorithms"], bool)
    assert report["content_sha256"] == sha256_json(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
