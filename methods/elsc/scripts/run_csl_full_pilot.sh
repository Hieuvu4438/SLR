#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

gain_gate=artifacts/campaign/csl_min_vs_base_b512_dev_3seed_gate_g.json
mechanism_gate=artifacts/campaign/csl_min_controls_b512_dev_3seed_gate_m.json
cross_dataset_gate=artifacts/campaign/csl_gate_x_3seed.json
baseline_run=runs/csl_base_b512_s42
min_run=runs/csl_min_b512_s42
min_cache=artifacts/cache/csl_daily_min_b512_s42_v1
full_cache=artifacts/cache/csl_daily_full_b512_s42_v1
full_config=artifacts/campaign/csl_full_b512_s42.yaml
continued_config=artifacts/campaign/csl_continued_min_b512_s42.yaml
full_run=runs/csl_full_b512_s42
continued_run=runs/csl_continued_min_b512_s42
full_gate=artifacts/campaign/csl_full_b512_s42_gate_f.json
preflight=artifacts/preflight/csl_full_b512_s42.json
log_path=artifacts/logs/csl_full_pilot_s42.log
hard_free_disk_gib=20
launch_free_disk_gib=28
gate_wait_timeout_seconds=172800
poll_seconds=60

mkdir -p artifacts/campaign artifacts/preflight artifacts/logs
printf '%s full_queue_start pid=%s git_head=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s full_queue_exit pid=%s status=%s test_accessed=false\n' \
    "$(date --iso-8601=seconds)" "$$" "$status" >>"$log_path"
}
trap campaign_exit EXIT

started_at=$(date +%s)
while [[ ! -f "$cross_dataset_gate" ]]; do
  now=$(date +%s)
  elapsed=$((now - started_at))
  if ! pgrep -f -- "run_csl_multiseed[.]sh" >/dev/null; then
    printf '%s multiseed_process_missing elapsed=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
    exit 1
  fi
  if (( elapsed >= gate_wait_timeout_seconds )); then
    printf '%s multiseed_wait_timeout elapsed=%s limit=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" "$gate_wait_timeout_seconds" >>"$log_path"
    exit 124
  fi
  printf '%s waiting_for_multiseed_gate elapsed=%s\n' \
    "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
  sleep "$poll_seconds"
done

gate_statuses=$(timeout 300 python - \
  "$gain_gate" "$mechanism_gate" "$cross_dataset_gate" <<'PY'
import json
import sys
from pathlib import Path

gain_path, mechanism_path, cross_path = map(Path, sys.argv[1:])
gain = json.loads(gain_path.read_text(encoding="utf-8"))
mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
cross = json.loads(cross_path.read_text(encoding="utf-8"))

for value, expected in ((gain, "G"), (mechanism, "M")):
    if value.get("schema_version") != 1 or value.get("gate") != expected:
        raise ValueError(f"invalid Gate {expected} artifact")
    if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
        raise ValueError(f"Gate {expected} violates dev-only isolation")
    if value.get("status") not in {"passed", "no_go", "insufficient_seeds"}:
        raise ValueError(f"Gate {expected} has an unknown status")

if (
    cross.get("schema_version") != 1
    or cross.get("gate") != "X"
    or cross.get("status") != "completed"
    or cross.get("dataset") != "csl_daily"
    or cross.get("split") != "dev"
    or cross.get("seeds") != [42, 1337, 2026]
    or cross.get("test_used_for_gate") is not False
):
    raise ValueError("invalid CSL Gate X scope artifact")
if cross.get("gain_gate_status") != gain["status"]:
    raise ValueError("Gate X/G status mismatch")
if cross.get("mechanism_gate_status") != mechanism["status"]:
    raise ValueError("Gate X/M status mismatch")
print(gain["status"], mechanism["status"])
PY
)
read -r gain_status mechanism_status <<<"$gate_statuses"
if [[ "$gain_status" != passed || "$mechanism_status" != passed ]]; then
  printf '%s full_skipped gate_g=%s gate_m=%s test_accessed=false\n' \
    "$(date --iso-8601=seconds)" "$gain_status" "$mechanism_status" >>"$log_path"
  exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  printf '%s blocked: tracked worktree is dirty\n' "$(date --iso-8601=seconds)" >>"$log_path"
  exit 2
fi

available_gib() {
  local available_bytes
  available_bytes=$(df --output=avail -B1 /home/haipd/SLR | tail -n 1 | tr -d ' ')
  printf '%s\n' "$((available_bytes / 1024 / 1024 / 1024))"
}

check_disk() {
  local available minimum_bytes
  available=$(df --output=avail -B1 /home/haipd/SLR | tail -n 1 | tr -d ' ')
  minimum_bytes=$((hard_free_disk_gib * 1024 * 1024 * 1024))
  if (( available < minimum_bytes )); then
    printf '%s disk_guard_failed available_bytes=%s minimum_bytes=%s\n' \
      "$(date --iso-8601=seconds)" "$available" "$minimum_bytes" >>"$log_path"
    return 1
  fi
  printf '%s disk_guard_passed available_bytes=%s minimum_bytes=%s\n' \
    "$(date --iso-8601=seconds)" "$available" "$minimum_bytes" >>"$log_path"
}

launch_available_gib=$(available_gib)
if (( launch_available_gib < launch_free_disk_gib )); then
  printf '%s full_deferred_disk available_gib=%s required_gib=%s\n' \
    "$(date --iso-8601=seconds)" "$launch_available_gib" "$launch_free_disk_gib" >>"$log_path"
  exit 75
fi

run_training() {
  local config=$1
  local run_dir=$2
  local label=$3
  check_disk
  if timeout 300 python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
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
  timeout 300 python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
    >>"$log_path" 2>&1
  printf '%s train_complete label=%s run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
}

evaluate_dev() {
  local run_dir=$1
  check_disk
  timeout 7200 python -m elsc.evaluate \
    --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
    >>"$log_path" 2>&1
}

report_pair() {
  local baseline=$1
  local method=$2
  local output=$3
  shift 3
  timeout 3600 python -m elsc.report \
    --baseline-runs "$baseline" --method-runs "$method" --split dev \
    --output "$output" "$@" >>"$log_path" 2>&1
}

check_disk
timeout 300 python -m elsc.provenance \
  --run-dir "$baseline_run" --config configs/csl_base.yaml >>"$log_path" 2>&1
timeout 300 python -m elsc.provenance \
  --run-dir "$min_run" --config artifacts/campaign/csl_min_b512_s42.yaml \
  >>"$log_path" 2>&1

timeout 300 python -m elsc.configure_stage \
  --template methods/elsc/configs/csl_full.yaml \
  --teacher-run "$baseline_run" \
  --student-run "$min_run" \
  --cache-path "$full_cache" \
  --output "$full_config" >>"$log_path" 2>&1
timeout 300 python -m elsc.configure_stage \
  --template methods/elsc/configs/csl_continued_min.yaml \
  --teacher-run "$baseline_run" \
  --student-run "$min_run" \
  --cache-path "$min_cache" \
  --output "$continued_config" >>"$log_path" 2>&1
timeout 300 python - "$continued_config" "$full_config" <<'PY' >>"$log_path" 2>&1
import json
import sys

from elsc.config import load_config
from elsc.report import configured_optimization_budget
from elsc.utils import sha256_json

reference = configured_optimization_budget(load_config(sys.argv[1], stage="train"))
candidate = configured_optimization_budget(load_config(sys.argv[2], stage="train"))
if reference != candidate:
    raise ValueError("CSL Full and continued-Min optimization budgets differ")
print(json.dumps({"optimization_budget": "matched", "sha256": sha256_json(reference)}))
PY

check_disk
timeout 43200 python -m elsc.mining.build_cache \
  --config "$full_config" --split train --device cuda:0 >>"$log_path" 2>&1
timeout 3600 python -m elsc.gate --gate F \
  --gain-gate "$gain_gate" \
  --mechanism-gate "$mechanism_gate" \
  --config "$full_config" \
  --cache "$full_cache" \
  --output "$full_gate" >>"$log_path" 2>&1

timeout 300 python - "$full_gate" <<'PY' >>"$log_path" 2>&1
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    value.get("schema_version") != 1
    or value.get("gate") != "F"
    or value.get("status") != "passed"
    or value.get("split") != "dev"
    or value.get("test_used_for_gate") is not False
):
    raise ValueError("CSL Gate F did not pass its RF/cache contract")
print(json.dumps({"gate": "F", "status": "passed", "test_accessed": False}))
PY

check_disk
timeout 1800 python -m elsc.training_preflight \
  --config "$full_config" \
  --batch-size 512 \
  --min-free-after-gib 4 \
  --output "$preflight" \
  --device cuda:0 >>"$log_path" 2>&1

run_training "$continued_config" "$continued_run" continued_min_step_matched_s42
evaluate_dev "$continued_run"
run_training "$full_config" "$full_run" full_s42
evaluate_dev "$full_run"

report_pair \
  "$continued_run" "$full_run" \
  artifacts/campaign/csl_full_vs_continued_min_b512_dev_s42.json \
  --require-matched-training-budget
report_pair \
  "$min_run" "$full_run" \
  artifacts/campaign/csl_full_vs_min_b512_dev_s42.json
report_pair \
  "$baseline_run" "$full_run" \
  artifacts/campaign/csl_full_vs_base_b512_dev_s42.json

printf '%s csl_full_pilot_complete seed=42 test_accessed=false\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
