"""Detached C27 prerequisite job: restore the registered PH adapted features.

The independent UniFormerV2 worker is paused only during GPU extraction and
resumed on success, failure, timeout, or supervisor loss via the C26 watchdog.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_c26_with_pause as safety

RUN_ID = "c27-ph-feature-recovery-001"
JOB = ROOT / "artifacts/slret_dataset_first/jobs" / RUN_ID
CHECKPOINT = ROOT / "artifacts/pretrained/rtmpose-l_384x288_c27.pth"
CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
)
CHECKPOINT_SHA256 = "13ce77ad08808333e2d4f850c632ac6068c837108e15f1c3902dcbcf30842db9"
EXTRACTOR = ROOT / "research/slret_goal/tools/seds_adapted_features.py"
TIMEOUT_SECONDS = 7200
VRAM_LIMIT_BYTES = 20_000_000_000
EXPECTED = (("dev", 519), ("train", 7096))


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def run() -> int:
    if (JOB / "status.json").exists():
        raise FileExistsError("run ID already has status; refusing duplicate launch")
    JOB.mkdir(parents=True, exist_ok=True)
    features_root = JOB / "features"
    features_root.mkdir(exist_ok=False)
    started = safety.stamp()
    status = dict(run_id=RUN_ID, status="STARTING", stage="verify_inputs",
                  started_at=started, updated_at=started, ended_at=None,
                  pid=os.getpid(), worker_pid=None, completed=0, total=7615,
                  exit_code=None, output_path=str(JOB), resume_verified=False)
    safety.atomic_json(JOB / "status.json", status)
    safety.atomic_json(JOB / "launch.json", dict(
        run_id=RUN_ID, started_at=started, cwd=str(ROOT), environment=sys.executable,
        supervisor_pid=os.getpid(), timeout_seconds=TIMEOUT_SECONDS,
        gpu_budget_seconds=TIMEOUT_SECONDS, owned_vram_limit_bytes=VRAM_LIMIT_BYTES,
        command=[sys.executable, "-u", str(Path(__file__).resolve()), "--run"],
        stages=[{"split": split, "count": count} for split, count in EXPECTED],
        checkpoint_url=CHECKPOINT_URL, checkpoint_path=str(CHECKPOINT),
        checkpoint_sha256=CHECKPOINT_SHA256, extractor=str(EXTRACTOR),
        extractor_sha256=digest(EXTRACTOR),
        target_pid=safety.TARGET_PID, target_pgid=safety.TARGET_PGID,
        target_start_ticks=safety.TARGET_START_TICKS,
        target_command=list(safety.TARGET_COMMAND),
        test_loaded=False, resume_policy="finally_and_watchdog"))
    paused = False
    resumed = False
    watcher = None
    child = None
    error = None
    timeout_hit = False
    results = []
    child_code = None
    resume_error = None
    deadline = time.monotonic() + TIMEOUT_SECONDS

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"C27 supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if shutil.disk_usage(JOB).free < 24 * 1024**3:
            raise RuntimeError("need >=24 GiB free for two PH feature splits and reserve")
        if not CHECKPOINT.is_file():
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            partial = CHECKPOINT.with_suffix(".part")
            status.update(stage="download_pose_checkpoint", updated_at=safety.stamp())
            safety.atomic_json(JOB / "status.json", status)
            with (JOB / "run.log").open("ab") as log:
                log.write(f"{safety.stamp()} download official RTMPose checkpoint\n".encode())
                log.flush()
                command = ["curl", "--fail", "--location", "--retry", "4",
                           "--continue-at", "-", "--output", str(partial), CHECKPOINT_URL]
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                status.update(worker_pid=child.pid, updated_at=safety.stamp())
                safety.atomic_json(JOB / "status.json", status)
                child_code = child.wait(timeout=min(900, max(1, deadline - time.monotonic())))
                if child_code:
                    raise RuntimeError(f"checkpoint download exited {child_code}")
            if digest(partial) != CHECKPOINT_SHA256:
                raise ValueError("downloaded RTMPose checkpoint SHA256 mismatch")
            os.replace(partial, CHECKPOINT)
        if digest(CHECKPOINT) != CHECKPOINT_SHA256:
            raise ValueError("RTMPose checkpoint SHA256 mismatch")

        target = safety.verified_target()
        if target["state"] in ("T", "t"):
            raise RuntimeError("UniFormerV2 already stopped; C27 cannot own pause")
        parent_start = safety.proc_stat(os.getpid())["start_ticks"]
        watcher_command = [sys.executable, "-u", str(Path(safety.__file__).resolve()),
                           "--watchdog", str(os.getpid()), str(parent_start), str(JOB)]
        with (JOB / "watchdog.log").open("xb") as log:
            watcher = subprocess.Popen(watcher_command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        status.update(watchdog_pid=watcher.pid, stage="pause_uniformerv2",
                      updated_at=safety.stamp())
        safety.atomic_json(JOB / "status.json", status)
        os.kill(safety.TARGET_PID, signal.SIGSTOP)
        paused = True
        status.update(target_pause_state=safety.wait_state(True),
                      target_paused_at=safety.stamp(), status="RUNNING",
                      updated_at=safety.stamp())
        safety.atomic_json(JOB / "status.json", status)

        for split, count in EXPECTED:
            name = f"seds-adapted-{split}-c27-001"
            command = [sys.executable, "-u", str(EXTRACTOR), "--run-id", name,
                       "--split", split, "--batch-size", "8",
                       "--output-root", str(features_root),
                       "--pose-checkpoint", str(CHECKPOINT)]
            status.update(stage=f"extract_{split}", worker_pid=None,
                          updated_at=safety.stamp())
            safety.atomic_json(JOB / "status.json", status)
            with (JOB / "run.log").open("ab") as log:
                log.write(f"\n{safety.stamp()} START {split}: {' '.join(command)}\n".encode())
                log.flush()
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                status.update(worker_pid=child.pid, updated_at=safety.stamp())
                safety.atomic_json(JOB / "status.json", status)
                while child.poll() is None:
                    if time.monotonic() > deadline:
                        timeout_hit = True
                        safety.stop_child(child)
                        raise TimeoutError("C27 feature-recovery wall cap exceeded")
                    report_path = features_root / name / "run.json"
                    if report_path.exists():
                        try:
                            report = json.loads(report_path.read_text())
                            prior = 0 if split == "dev" else 519
                            status.update(completed=prior + int(report.get("completed", 0)),
                                          stage=f"extract_{split}", updated_at=safety.stamp())
                            safety.atomic_json(JOB / "status.json", status)
                        except (OSError, ValueError):
                            pass
                    time.sleep(30)
                child_code = child.returncode
            report_path = features_root / name / "run.json"
            report = json.loads(report_path.read_text()) if report_path.is_file() else {}
            if child_code != 0 or report.get("status") != "completed" or report.get("completed") != count:
                raise RuntimeError(f"{split} extraction failed: exit={child_code}, report={report.get('status')}")
            if report.get("peak_cuda_bytes", VRAM_LIMIT_BYTES + 1) > VRAM_LIMIT_BYTES:
                raise RuntimeError(f"{split} extraction exceeded C27 owned VRAM cap")
            results.append(dict(split=split, count=count, run_id=name,
                                output=str(features_root / name),
                                wall_seconds=report.get("wall_seconds"),
                                peak_cuda_bytes=report.get("peak_cuda_bytes")))
            status.update(completed=sum(row["count"] for row in results),
                          updated_at=safety.stamp())
            safety.atomic_json(JOB / "status.json", status)
    except BaseException:
        error = traceback.format_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        safety.stop_child(child)
        if paused:
            try:
                state = safety.verified_target()["state"]
                if state in ("T", "t"):
                    os.kill(safety.TARGET_PID, signal.SIGCONT)
                status["target_resume_state"] = safety.wait_state(False)
                resumed = True
            except BaseException:
                resume_error = traceback.format_exc()
        final = "TIMED_OUT" if timeout_hit else ("COMPLETED" if error is None else "FAILED")
        status.update(status=final, stage="done", ended_at=safety.stamp(),
                      updated_at=safety.stamp(), exit_code=0 if final == "COMPLETED" else 1,
                      worker_exit_code=child_code, resume_verified=resumed,
                      target_resume_error=resume_error)
        safety.atomic_json(JOB / "status.json", status)
        safety.atomic_json(JOB / "summary.json", dict(
            run_id=RUN_ID, status=final, exit_code=status["exit_code"],
            worker_exit_code=child_code, timed_out=timeout_hit,
            target_paused=paused, target_resumed=resumed,
            target_resume_error=resume_error, error=error,
            stages=results, checkpoint=str(CHECKPOINT),
            checkpoint_sha256=CHECKPOINT_SHA256, test_loaded=False,
            ended_at=safety.stamp(), output_path=str(JOB)))
        with (JOB / "run.log").open("ab") as log:
            log.write(f"\n{safety.stamp()} SUPERVISOR status={final} "
                      f"uniformerv2_resumed={resumed}\n".encode())
            if error:
                log.write(error.encode())
    return 0 if final == "COMPLETED" else 1


def launch() -> None:
    JOB.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run"]
    with (JOB / "launcher.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print(json.dumps(dict(run_id=RUN_ID, pid=process.pid,
                          status=str(JOB / "status.json"),
                          log=str(JOB / "run.log")), sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] == ["--launch"]:
        launch()
    elif sys.argv[1:] == ["--run"]:
        raise SystemExit(run())
    else:
        raise SystemExit("use --launch or --run")
