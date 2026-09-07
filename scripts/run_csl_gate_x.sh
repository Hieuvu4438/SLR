#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

base_config=configs/csl_base.yaml
base_run=runs/csl_base_b512_s42
min_template=configs/csl_min.yaml
min_config=artifacts/campaign/csl_min_b512_s42.yaml
min_cache=artifacts/cache/csl_daily_min_b512_s42_v1
min_run=runs/csl_min_b512_s42
comparison=artifacts/campaign/csl_min_vs_base_b512_dev_s42.json
gate_report=artifacts/campaign/csl_min_vs_base_b512_dev_s42_gate_g.json
asset_report=artifacts/transfer/csl_base_asset_audit.json
checkpoint_report=artifacts/parity/csl_checkpoint_smoke.json
agnostic_report=artifacts/features_reextracted/csl_domain_agnostic/extraction_report_train_dev.json
aware_report=artifacts/features_reextracted/csl_domain_aware_h2s_transfer/extraction_report_train_dev.json
log_path=artifacts/logs/csl_gate_x_s42.log
wait_timeout_seconds=108000
poll_seconds=60
minimum_free_disk_gib=20

mkdir -p artifacts/campaign artifacts/logs artifacts/parity artifacts/transfer
printf '%s gate_x_queue_start pid=%s git_head=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s gate_x_queue_exit pid=%s status=%s test_accessed=false\n' \
    "$(date --iso-8601=seconds)" "$$" "$status" >>"$log_path"
}
trap campaign_exit EXIT

check_disk() {
  local available_bytes minimum_bytes
  available_bytes=$(df --output=avail -B1 /home/haipd/SLR | tail -n 1 | tr -d ' ')
  minimum_bytes=$((minimum_free_disk_gib * 1024 * 1024 * 1024))
  if (( available_bytes < minimum_bytes )); then
    printf '%s disk_guard_failed available_bytes=%s minimum_bytes=%s\n' \
      "$(date --iso-8601=seconds)" "$available_bytes" "$minimum_bytes" >>"$log_path"
    return 1
  fi
  printf '%s disk_guard_passed available_bytes=%s minimum_bytes=%s\n' \
    "$(date --iso-8601=seconds)" "$available_bytes" "$minimum_bytes" >>"$log_path"
}

extraction_complete() {
  python - "$agnostic_report" "$aware_report" <<'PY' >>"$log_path" 2>&1
import json
import sys
from pathlib import Path

expected = {
    "domain_agnostic": "6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f",
    "domain_aware_h2s_transfer": "99e101d696ff63131b5d44fa6e465201216604ba5d8cc773f3cefa4a96ebd518",
}
reports = []
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.is_file():
        raise SystemExit(3)
    value = json.loads(path.read_text(encoding="utf-8"))
    reports.append(value)
    if value.get("failed"):
        raise SystemExit(2)
    if value.get("status") != "complete":
        raise SystemExit(3)
    if value.get("splits") != ["train", "dev"] or value.get("videos") != 19478:
        raise SystemExit(2)
    if value.get("completed") != 19478:
        raise SystemExit(2)
    stream = value.get("stream_name")
    if value.get("checkpoint_sha256") != expected.get(stream):
        raise SystemExit(2)
if {item["stream_name"] for item in reports} != set(expected):
    raise SystemExit(2)
if {item["recipe_sha256"] for item in reports} != {
    "50045403867d7bdbee644de13e94389b5567c67b0f7f87e7218952dcde49a571"
}:
    raise SystemExit(2)
print(json.dumps({"status": "complete", "videos_per_stream": 19478, "test_accessed": False}))
PY
}

started_at=$(date +%s)
while true; do
  if extraction_complete; then
    break
  else
    extraction_status=$?
  fi
  if (( extraction_status == 2 )); then
    printf '%s extraction_validation_failed\n' "$(date --iso-8601=seconds)" >>"$log_path"
    exit 1
  fi
  now=$(date +%s)
  elapsed=$((now - started_at))
  if ! pgrep -f -- "python -m elsc.features.i3d.*csl_domain" >/dev/null; then
    printf '%s extraction_process_missing elapsed=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
    exit 1
  fi
  if (( elapsed >= wait_timeout_seconds )); then
    printf '%s extraction_wait_timeout elapsed=%s limit=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" "$wait_timeout_seconds" >>"$log_path"
    exit 124
  fi
  printf '%s waiting_for_extraction elapsed=%s\n' \
    "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
  sleep "$poll_seconds"
done

if ! git diff --quiet || ! git diff --cached --quiet; then
  printf '%s blocked: tracked worktree is dirty\n' "$(date --iso-8601=seconds)" >>"$log_path"
  exit 2
fi

check_disk
timeout 7200 python -m elsc.prepare --config "$base_config" --splits train dev \
  >>"$log_path" 2>&1
timeout 3600 python -m elsc.audit --config "$base_config" --stage assets \
  --output "$asset_report" >>"$log_path" 2>&1
timeout 3600 python -m elsc.audit --config "$base_config" --stage checkpoint \
  --output "$checkpoint_report" >>"$log_path" 2>&1

run_training() {
  local config=$1
  local run_dir=$2
  local label=$3
  check_disk
  if python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
    >>"$log_path" 2>&1; then
    printf '%s run_already_complete label=%s run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
    return 0
  fi
  if [[ -f "$run_dir/selection.json" || -f "$run_dir/run_summary.json" ]]; then
    printf '%s invalid_existing_run label=%s run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
    return 1
  fi
  local args=(--config "$config" --run-dir "$run_dir" --device cuda:0)
  if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
    args+=(--resume "$run_dir/checkpoints/last.pt")
  fi
  printf '%s train_start label=%s run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
  timeout 43200 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m elsc.train "${args[@]}" >>"$log_path" 2>&1
  python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
    >>"$log_path" 2>&1
  printf '%s train_complete label=%s run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
}

run_training "$base_config" "$base_run" baseline
timeout 7200 python -m elsc.evaluate \
  --run-dir "$base_run" --split dev --checkpoint best_dev --device cuda:0 \
  >>"$log_path" 2>&1

python -m elsc.configure_stage \
  --template "$min_template" \
  --teacher-run "$base_run" \
  --cache-path "$min_cache" \
  --output "$min_config" >>"$log_path" 2>&1
check_disk
timeout 43200 python -m elsc.mining.build_cache \
  --config "$min_config" --split train --device cuda:0 >>"$log_path" 2>&1
run_training "$min_config" "$min_run" elsc_min
timeout 7200 python -m elsc.evaluate \
  --run-dir "$min_run" --split dev --checkpoint best_dev --device cuda:0 \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "$base_run" --method-runs "$min_run" --split dev \
  --output "$comparison" >>"$log_path" 2>&1

gate_status=0
python -m elsc.gate --gate G --report "$comparison" --output "$gate_report" \
  >>"$log_path" 2>&1 || gate_status=$?
python - "$gate_report" "$gate_status" <<'PY' >>"$log_path" 2>&1
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
status = value.get("status")
exit_code = int(sys.argv[2])
if value.get("gate") != "G" or status not in {"passed", "no_go", "insufficient_seeds"}:
    raise ValueError("invalid CSL Gate G artifact")
if (status == "passed") != (exit_code == 0):
    raise ValueError("CSL Gate G status and CLI exit code disagree")
print(json.dumps({"gate": "G", "status": status, "test_accessed": False}))
PY

printf '%s gate_x_seed42_complete gate_g_cli_status=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$gate_status" >>"$log_path"
