"""Detached C26 supervisor: stop only the verified UniFormerV2 GPU worker.

Normal exits, worker errors, SIGTERM/SIGINT and hard timeout all attempt to
resume the exact worker. An independent local watchdog covers supervisor death.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "c26-unisign-csl-pilot-002"
JOB = ROOT / "artifacts/slret_goal/jobs" / RUN_ID
WORKER = ROOT / "research/slret_dataset_first/tools/run_c26_pilot.py"
FEATURE_SOURCE = ROOT / "artifacts/slret_goal/jobs/c26-unisign-csl-pilot-001/features"
TIMEOUT_SECONDS = 900
TARGET_PID = 3493585
TARGET_PGID = 3493571
TARGET_START_TICKS = 190163677
TARGET_COMMAND = (
    "/home/haipd/miniconda3/envs/seds/bin/python", "-u",
    "scripts/extract_ph_uniformerv2.py", "--splits", "test", "dev", "train",
    "--stride", "1", "--batch-size", "8", "--device", "cuda:0",
)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def proc_stat(pid: int) -> dict:
    raw = (Path("/proc") / str(pid) / "stat").read_text()
    # The comm field can contain spaces; all remaining fields start after ") ".
    fields = raw.rsplit(") ", 1)[1].split()
    return {"state": fields[0], "pgid": int(fields[2]),
            "start_ticks": int(fields[19])}


def cmdline(pid: int) -> tuple[str, ...]:
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return tuple(item.decode() for item in raw.split(b"\0") if item)


def verified_target() -> dict:
    path = Path("/proc") / str(TARGET_PID)
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise RuntimeError("UniFormerV2 target is absent or not owned by this user")
    result = proc_stat(TARGET_PID)
    if (result["pgid"] != TARGET_PGID or result["start_ticks"] != TARGET_START_TICKS
            or cmdline(TARGET_PID) != TARGET_COMMAND):
        raise RuntimeError("UniFormerV2 target identity changed; refusing signal")
    return result


def wait_state(stopped: bool, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = verified_target()["state"]
        if (state in ("T", "t")) == stopped:
            return state
        time.sleep(0.1)
    raise RuntimeError(f"UniFormerV2 stop/resume state not verified: {verified_target()['state']}")


def parent_alive(pid: int, start_ticks: int) -> bool:
    try:
        state = proc_stat(pid)
        return state["start_ticks"] == start_ticks and state["state"] not in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError):
        return False


def watchdog(parent_pid: int, parent_start_ticks: int, job_dir: Path) -> None:
    while parent_alive(parent_pid, parent_start_ticks):
        time.sleep(10)
    record = {"observed_at": stamp(), "parent_pid": parent_pid,
              "target_pid": TARGET_PID, "action": "none"}
    try:
        state = verified_target()["state"]
        if state in ("T", "t"):
            os.kill(TARGET_PID, signal.SIGCONT)
            record.update(action="SIGCONT", resume_state=wait_state(False))
    except (FileNotFoundError, ProcessLookupError):
        record["target_status"] = "gone"
    except BaseException:
        record["error"] = traceback.format_exc()
    atomic_json(job_dir / "watchdog_recovery.json", record)
    status_path = job_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            if status.get("status") in ("STARTING", "RUNNING"):
                status.update(status="INTERRUPTED", stage="supervisor_lost",
                              ended_at=stamp(), updated_at=stamp(),
                              resume_verified=record.get("action") == "SIGCONT"
                              or record.get("action") == "none")
                atomic_json(status_path, status)
        except BaseException:
            pass


def stop_child(child: subprocess.Popen | None) -> None:
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=15)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run(job_dir: Path) -> int:
    if (job_dir / "status.json").exists():
        raise FileExistsError("C26 run ID already has status; refusing duplicate launch")
    job_dir.mkdir(parents=True, exist_ok=True)
    started = stamp()
    worker_command = [sys.executable, "-u", str(WORKER), "--job-dir", str(job_dir),
                      "--feature-source", str(FEATURE_SOURCE)]
    status = {"run_id": RUN_ID, "status": "STARTING", "stage": "verify_target",
              "started_at": started, "updated_at": started, "ended_at": None,
              "pid": os.getpid(), "worker_pid": None, "completed": 0,
              "total": 18401 + 1077, "exit_code": None,
              "output_path": str(job_dir), "resume_verified": False}
    atomic_json(job_dir / "status.json", status)
    atomic_json(job_dir / "launch.json", {
        "run_id": RUN_ID, "started_at": started, "cwd": str(ROOT),
        "environment": sys.executable, "supervisor_pid": os.getpid(),
        "command": worker_command, "timeout_seconds": TIMEOUT_SECONDS,
        "gpu_budget_seconds": TIMEOUT_SECONDS, "owned_vram_limit_bytes": 20_000_000_000,
        "train_examples": 18401, "dev_examples": 1077, "max_pose_frames": 256,
        "feature_batch_size": 8, "feature_shard_size": 128,
        "feature_source": str(FEATURE_SOURCE),
        "technical_retry_of": "c26-unisign-csl-pilot-001",
        "projection_epochs": 5, "projection_batch_size": 256, "seed": 42,
        "target_pid": TARGET_PID, "target_start_ticks": TARGET_START_TICKS,
        "target_pgid": TARGET_PGID, "target_command": list(TARGET_COMMAND),
        "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        "test_loaded": False, "resume_policy": "finally_and_watchdog",
    })
    paused = False
    child: subprocess.Popen | None = None
    watcher: subprocess.Popen | None = None
    timed_out = False
    error: str | None = None
    worker_summary: dict | None = None
    worker_code: int | None = None
    resume_error: str | None = None
    resumed = False

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"C26 supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        target = verified_target()
        if target["state"] in ("T", "t"):
            raise RuntimeError("UniFormerV2 was stopped before C26; refusing ownership")
        parent_start = proc_stat(os.getpid())["start_ticks"]
        watcher_command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--watchdog", str(os.getpid()), str(parent_start), str(job_dir)]
        with (job_dir / "watchdog.log").open("xb") as log:
            watcher = subprocess.Popen(watcher_command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        status.update(watchdog_pid=watcher.pid, stage="pause_uniformerv2", updated_at=stamp())
        atomic_json(job_dir / "status.json", status)

        os.kill(TARGET_PID, signal.SIGSTOP)
        paused = True
        status.update(target_pause_state=wait_state(True), target_paused_at=stamp(),
                      stage="start_worker", updated_at=stamp())
        atomic_json(job_dir / "status.json", status)
        with (job_dir / "run.log").open("xb") as log:
            child = subprocess.Popen(worker_command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            status.update(status="RUNNING", stage="load_donor", worker_pid=child.pid,
                          updated_at=stamp())
            atomic_json(job_dir / "status.json", status)
            try:
                worker_code = child.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_child(child)
                worker_code = child.returncode
        if worker_code != 0:
            raise RuntimeError(f"C26 worker exited {worker_code}")
        summary_path = job_dir / "worker_summary.json"
        if not summary_path.exists():
            raise RuntimeError("C26 worker exit 0 but summary is missing")
        worker_summary = json.loads(summary_path.read_text())
        if worker_summary.get("status") != "COMPLETED" or worker_summary.get("test_loaded") is not False:
            raise RuntimeError("C26 worker summary failed terminal validation")
    except BaseException:
        error = traceback.format_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_child(child)
        if paused:
            try:
                state = verified_target()["state"]
                if state in ("T", "t"):
                    os.kill(TARGET_PID, signal.SIGCONT)
                status["target_resume_state"] = wait_state(False)
                resumed = True
            except BaseException:
                resume_error = traceback.format_exc()
        if not resumed and paused:
            error = (error or "") + "\nUniFormerV2 resume failed: " + str(resume_error)
        final = "TIMED_OUT" if timed_out else ("COMPLETED" if error is None else "FAILED")
        status = json.loads((job_dir / "status.json").read_text())
        status.update(status=final, stage="done", ended_at=stamp(), updated_at=stamp(),
                      exit_code=0 if final == "COMPLETED" else 1,
                      worker_exit_code=worker_code, resume_verified=resumed,
                      target_resume_error=resume_error)
        atomic_json(job_dir / "status.json", status)
        summary = {"run_id": RUN_ID, "status": final, "exit_code": status["exit_code"],
                   "worker_exit_code": worker_code, "timed_out": timed_out,
                   "target_paused": paused, "target_resumed": resumed,
                   "target_resume_error": resume_error, "error": error,
                   "worker_summary": worker_summary, "ended_at": stamp(),
                   "output_path": str(job_dir)}
        atomic_json(job_dir / "summary.json", summary)
        with (job_dir / "run.log").open("ab") as log:
            log.write((f"\n{stamp()} SUPERVISOR status={final} worker_exit={worker_code} "
                       f"uniformerv2_resumed={resumed}\n").encode())
            if error:
                log.write(error.encode())
        if watcher is not None and watcher.poll() is not None and not resumed:
            print("C26 watchdog ended before UniFormerV2 resume", flush=True)
    return 0 if final == "COMPLETED" else 1


def launch() -> None:
    JOB.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--job-dir", str(JOB)]
    with (JOB / "launcher.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print(json.dumps({"run_id": RUN_ID, "pid": process.pid,
                      "status": str(JOB / "status.json"),
                      "log": str(JOB / "run.log")}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--watchdog", nargs=3)
    args = parser.parse_args()
    if args.launch:
        launch()
        return 0
    if args.watchdog:
        pid, start, directory = args.watchdog
        watchdog(int(pid), int(start), Path(directory))
        return 0
    if args.job_dir:
        return run(args.job_dir)
    parser.error("provide --launch or --job-dir")


if __name__ == "__main__":
    raise SystemExit(main())
