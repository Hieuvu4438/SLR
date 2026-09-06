from __future__ import annotations

from pathlib import Path

import pytest
import torch

from elsc.resources import ResourceGuardError, require_resources, require_storage_budget


def test_disk_guard_fails_before_large_artifact_generation(tmp_path: Path):
    with pytest.raises(ResourceGuardError, match="free disk"):
        require_resources(
            tmp_path,
            torch.device("cpu"),
            min_disk_gib=10**9,
            min_gpu_gib=0,
            operation="test operation",
        )


def test_cpu_resource_snapshot_does_not_require_gpu(tmp_path: Path):
    result = require_resources(
        tmp_path,
        torch.device("cpu"),
        min_disk_gib=0,
        min_gpu_gib=10**9,
        operation="cpu test",
    )
    assert result["gpu_free_bytes"] is None


def test_storage_budget_accounts_for_planned_writes(tmp_path: Path):
    with pytest.raises(ResourceGuardError, match="plans to write"):
        require_storage_budget(
            tmp_path,
            planned_write_bytes=10**30,
            min_remaining_gib=1,
            operation="feature test",
        )

    result = require_storage_budget(
        tmp_path,
        planned_write_bytes=1,
        min_remaining_gib=0,
        operation="feature test",
    )
    assert result["projected_disk_free_bytes"] == result["disk_free_bytes"] - 1
