from __future__ import annotations

import subprocess
from pathlib import Path

from elsc.utils import git_worktree_state


def test_git_worktree_state_records_commit_and_tracked_dirty_state(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "ELSC Test"], check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)

    clean = git_worktree_state(tracked)
    assert clean["status"] == "ready"
    assert len(clean["commit"]) == 40
    assert clean["tracked_worktree_dirty"] is False

    tracked.write_text("changed\n", encoding="utf-8")
    dirty = git_worktree_state(tracked)
    assert dirty["commit"] == clean["commit"]
    assert dirty["tracked_worktree_dirty"] is True
