#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

video_root=/home/dongvk/datasets/CSL_Daily_Sentence_Crop/videos
train_list=artifacts/transfer/csl_daily_video_lists/train.txt
dev_list=artifacts/transfer/csl_daily_video_lists/dev.txt
dry_run_report=artifacts/features_reextracted/csl_domain_agnostic/extraction_report_train_dev.json
temporal_root=artifacts/features_reextracted/csl_temporal_metadata
agnostic_root=artifacts/features_reextracted/csl_domain_agnostic
aware_root=artifacts/features_reextracted/csl_domain_aware_h2s_transfer
agnostic_checkpoint=artifacts/pretrained/bsl5k.pth.tar
aware_checkpoint=artifacts/pretrained/domain_aware_I3D_H2S.pth.tar
agnostic_sha=6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f
aware_sha=99e101d696ff63131b5d44fa6e465201216604ba5d8cc773f3cefa4a96ebd518
log_root=artifacts/logs
launcher_log=$log_root/csl_i3d_train_dev_launcher.log
reserve_gib=32
timeout_seconds=86400

mkdir -p "$log_root"
if ! git diff --quiet || ! git diff --cached --quiet; then
  printf '%s blocked: tracked worktree is dirty\n' "$(date --iso-8601=seconds)" \
    >>"$launcher_log"
  exit 2
fi

python - "$dry_run_report" "$train_list" "$dev_list" "$reserve_gib" <<'PY' \
  >>"$launcher_log"
import json
import shutil
import sys
from pathlib import Path

from elsc.utils import sha256_file

report_path, train_path, dev_path = map(Path, sys.argv[1:4])
reserve_gib = float(sys.argv[4])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "planned" or report.get("splits") != ["train", "dev"]:
    raise SystemExit("dry-run report is not a train+dev extraction plan")
expected = report.get("split_video_lists", {})
for split, path in (("train", train_path), ("dev", dev_path)):
    if expected.get(split, {}).get("sha256") != sha256_file(path):
        raise SystemExit(f"{split} video list differs from the measured dry-run plan")
planned = 2 * int(report["planned_write_bytes_with_atomic_margin"])
free = shutil.disk_usage(".").free
reserve = int(reserve_gib * 1024**3)
if free - planned < reserve:
    raise SystemExit(
        f"combined extraction would leave {(free-planned)/1024**3:.2f} GiB; "
        f"required reserve is {reserve_gib:g} GiB"
    )
print(
    json.dumps(
        {
            "status": "ready",
            "free_gib": free / 1024**3,
            "combined_planned_gib": planned / 1024**3,
            "projected_free_gib": (free - planned) / 1024**3,
            "reserve_gib": reserve_gib,
            "splits": ["train", "dev"],
            "test_accessed": False,
        },
        sort_keys=True,
    )
)
PY

common_args=(
  --video-root "$video_root"
  --temporal-metadata-root "$temporal_root"
  --splits train dev
  --split-video-list "train=$train_list"
  --split-video-list "dev=$dev_list"
  --batch-size 128
  --device cuda:0
  --min-free-disk-gib "$reserve_gib"
  --min-free-gpu-gib 8
  --report-interval 25
)

printf '%s extraction_start git_commit=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" >>"$launcher_log"

timeout "$timeout_seconds" python -m elsc.features.i3d \
  "${common_args[@]}" \
  --checkpoint "$agnostic_checkpoint" \
  --expected-checkpoint-sha256 "$agnostic_sha" \
  --stream-name domain_agnostic \
  --output-root "$agnostic_root" \
  >>"$log_root/csl_i3d_agnostic_train_dev.log" 2>&1 &
agnostic_pid=$!

timeout "$timeout_seconds" python -m elsc.features.i3d \
  "${common_args[@]}" \
  --checkpoint "$aware_checkpoint" \
  --expected-checkpoint-sha256 "$aware_sha" \
  --stream-name domain_aware_h2s_transfer \
  --output-root "$aware_root" \
  >>"$log_root/csl_i3d_aware_train_dev.log" 2>&1 &
aware_pid=$!

set +e
wait "$agnostic_pid"
agnostic_status=$?
wait "$aware_pid"
aware_status=$?
set -e

printf '%s extraction_end agnostic_status=%s aware_status=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$agnostic_status" "$aware_status" >>"$launcher_log"
if (( agnostic_status != 0 || aware_status != 0 )); then
  exit 1
fi
