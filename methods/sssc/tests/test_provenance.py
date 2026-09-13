from __future__ import annotations

from pathlib import Path

from method1.provenance import implementation_source_report, runtime_environment_report
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


def test_implementation_source_report_changes_with_semantic_source(tmp_path: Path) -> None:
    package = tmp_path / "method1"
    modules = tmp_path / "upret" / "modules"
    package.mkdir()
    modules.mkdir(parents=True)
    local_source = package / "train.py"
    local_source.write_text("VALUE = 1\n", encoding="utf-8")
    (modules / "modeling.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = implementation_source_report(
        method_package_root=package,
        upret_root=tmp_path / "upret",
        supporting_files=[],
    )
    local_source.write_text("VALUE = 3\n", encoding="utf-8")
    second = implementation_source_report(
        method_package_root=package,
        upret_root=tmp_path / "upret",
        supporting_files=[],
    )
    assert first["content_sha256"] != second["content_sha256"]
    assert set(first["files"]) == {"method1/train.py", "upret/modules/modeling.py"}
