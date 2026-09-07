#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

log_path=artifacts/logs/csl_multiseed.log
seed42_gate=artifacts/campaign/csl_min_vs_base_b512_dev_s42_gate_g.json
minimum_free_disk_gib=20
gate_wait_timeout_seconds=172800
poll_seconds=60
mkdir -p artifacts/campaign artifacts/logs

printf '%s multiseed_queue_start pid=%s git_head=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s multiseed_queue_exit pid=%s status=%s test_accessed=false\n' \
    "$(date --iso-8601=seconds)" "$$" "$status" >>"$log_path"
}
trap campaign_exit EXIT

started_at=$(date +%s)
while [[ ! -f "$seed42_gate" ]]; do
  now=$(date +%s)
  elapsed=$((now - started_at))
  if ! pgrep -f -- "run_csl_gate_x[.]sh" >/dev/null; then
    printf '%s seed42_gate_process_missing elapsed=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
    exit 1
  fi
  if (( elapsed >= gate_wait_timeout_seconds )); then
    printf '%s seed42_gate_wait_timeout elapsed=%s limit=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" "$gate_wait_timeout_seconds" >>"$log_path"
    exit 124
  fi
  printf '%s waiting_for_seed42_gate elapsed=%s\n' \
    "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
  sleep "$poll_seconds"
done

seed42_status=$(python - "$seed42_gate" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    value.get("schema_version") != 1
    or value.get("gate") != "G"
    or value.get("split") != "dev"
    or value.get("test_used_for_gate") is not False
):
    raise ValueError("invalid seed-42 CSL Gate G artifact")
status = value.get("status")
if status not in {"passed", "no_go", "insufficient_seeds"}:
    raise ValueError(f"unknown seed-42 Gate G status: {status}")
print(status)
PY
)
if [[ "$seed42_status" != passed ]]; then
  printf '%s expansion_skipped seed42_gate_g=%s test_accessed=false\n' \
    "$(date --iso-8601=seconds)" "$seed42_status" >>"$log_path"
  exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  printf '%s blocked: tracked worktree is dirty\n' "$(date --iso-8601=seconds)" >>"$log_path"
  exit 2
fi

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

evaluate_dev() {
  local run_dir=$1
  check_disk
  timeout 7200 python -m elsc.evaluate \
    --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
    >>"$log_path" 2>&1
}

configure_stage() {
  local template=$1
  local teacher_run=$2
  local cache_path=$3
  local output=$4
  local args=(--template "$template" --teacher-run "$teacher_run" --output "$output")
  if [[ -n "$cache_path" ]]; then
    args+=(--cache-path "$cache_path")
  fi
  python -m elsc.configure_stage "${args[@]}" >>"$log_path" 2>&1
}

report_pair() {
  local baseline_run=$1
  local method_run=$2
  local output=$3
  timeout 3600 python -m elsc.report \
    --baseline-runs "$baseline_run" --method-runs "$method_run" --split dev \
    --output "$output" >>"$log_path" 2>&1
}

run_controls() {
  local seed=$1
  local baseline_run=$2
  local min_run=$3
  local cache_path=$4
  local control_name template control_config control_run
  for control_name in adapter_only caption random_span; do
    case "$control_name" in
      adapter_only) template=methods/elsc/configs/csl_adapter_only.yaml ;;
      caption) template=methods/elsc/configs/csl_caption.yaml ;;
      random_span) template=methods/elsc/configs/csl_random_span.yaml ;;
    esac
    control_config="artifacts/campaign/csl_${control_name}_b512_s${seed}.yaml"
    control_run="runs/csl_${control_name}_b512_s${seed}"
    if [[ "$control_name" == adapter_only ]]; then
      configure_stage "$template" "$baseline_run" "" "$control_config"
    else
      configure_stage "$template" "$baseline_run" "$cache_path" "$control_config"
    fi
    run_training "$control_config" "$control_run" "${control_name}_s${seed}"
    evaluate_dev "$control_run"
    report_pair \
      "$baseline_run" "$control_run" \
      "artifacts/campaign/csl_${control_name}_vs_base_b512_dev_s${seed}.json"
    report_pair \
      "$control_run" "$min_run" \
      "artifacts/campaign/csl_min_vs_${control_name}_b512_dev_s${seed}.json"
  done
}

run_seed() {
  local seed=$1
  local baseline_config="configs/csl_base_s${seed}.yaml"
  local baseline_run="runs/csl_base_b512_s${seed}"
  local cache_path="artifacts/cache/csl_daily_min_b512_s${seed}_v1"
  local min_config="artifacts/campaign/csl_min_b512_s${seed}.yaml"
  local min_run="runs/csl_min_b512_s${seed}"

  run_training "$baseline_config" "$baseline_run" "baseline_s${seed}"
  evaluate_dev "$baseline_run"
  configure_stage methods/elsc/configs/csl_min.yaml "$baseline_run" "$cache_path" "$min_config"
  check_disk
  timeout 43200 python -m elsc.mining.build_cache \
    --config "$min_config" --split train --device cuda:0 >>"$log_path" 2>&1
  run_training "$min_config" "$min_run" "min_s${seed}"
  evaluate_dev "$min_run"
  report_pair \
    "$baseline_run" "$min_run" \
    "artifacts/campaign/csl_min_vs_base_b512_dev_s${seed}.json"
  run_controls "$seed" "$baseline_run" "$min_run" "$cache_path"
  printf '%s seed_complete seed=%s test_accessed=false\n' \
    "$(date --iso-8601=seconds)" "$seed" >>"$log_path"
}

run_controls \
  42 runs/csl_base_b512_s42 runs/csl_min_b512_s42 \
  artifacts/cache/csl_daily_min_b512_s42_v1
run_seed 1337
run_seed 2026

baseline_runs=(
  runs/csl_base_b512_s42
  runs/csl_base_b512_s1337
  runs/csl_base_b512_s2026
)
min_runs=(
  runs/csl_min_b512_s42
  runs/csl_min_b512_s1337
  runs/csl_min_b512_s2026
)
caption_runs=(
  runs/csl_caption_b512_s42
  runs/csl_caption_b512_s1337
  runs/csl_caption_b512_s2026
)
random_runs=(
  runs/csl_random_span_b512_s42
  runs/csl_random_span_b512_s1337
  runs/csl_random_span_b512_s2026
)

timeout 3600 python -m elsc.report \
  --baseline-runs "${baseline_runs[@]}" --method-runs "${min_runs[@]}" --split dev \
  --output artifacts/campaign/csl_min_vs_base_b512_dev_3seed.json \
  >>"$log_path" 2>&1
timeout 3600 python -m elsc.report \
  --baseline-runs "${caption_runs[@]}" --method-runs "${min_runs[@]}" --split dev \
  --output artifacts/campaign/csl_min_vs_caption_b512_dev_3seed.json \
  >>"$log_path" 2>&1
timeout 3600 python -m elsc.report \
  --baseline-runs "${random_runs[@]}" --method-runs "${min_runs[@]}" --split dev \
  --output artifacts/campaign/csl_min_vs_random_span_b512_dev_3seed.json \
  >>"$log_path" 2>&1

run_gate() {
  local expected_gate=$1
  local output=$2
  shift 2
  local pending_output="${output}.pending.$$"
  local cli_status=0
  "$@" --output "$pending_output" || cli_status=$?
  python - "$pending_output" "$expected_gate" "$cli_status" <<'PY' >>"$log_path" 2>&1
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
expected_gate = sys.argv[2]
exit_code = int(sys.argv[3])
status = value.get("status")
if value.get("schema_version") != 1 or value.get("gate") != expected_gate:
    raise ValueError("invalid gate artifact")
if status not in {"passed", "no_go", "insufficient_seeds"}:
    raise ValueError(f"unknown gate status: {status}")
if (status == "passed") != (exit_code == 0):
    raise ValueError("gate status and CLI exit code disagree")
print(json.dumps({"gate": expected_gate, "status": status, "test_accessed": False}))
PY
  mv -f -- "$pending_output" "$output"
}

run_gate G artifacts/campaign/csl_min_vs_base_b512_dev_3seed_gate_g.json \
  python -m elsc.gate --gate G \
  --report artifacts/campaign/csl_min_vs_base_b512_dev_3seed.json
run_gate M artifacts/campaign/csl_min_controls_b512_dev_3seed_gate_m.json \
  python -m elsc.gate --gate M \
  --true-vs-random-report artifacts/campaign/csl_min_vs_random_span_b512_dev_3seed.json \
  --true-vs-caption-report artifacts/campaign/csl_min_vs_caption_b512_dev_3seed.json

python - <<'PY'
import json
from pathlib import Path

from elsc.utils import atomic_json_dump

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

gain = load("artifacts/campaign/csl_min_vs_base_b512_dev_3seed_gate_g.json")
mechanism = load("artifacts/campaign/csl_min_controls_b512_dev_3seed_gate_m.json")
value = {
    "schema_version": 1,
    "gate": "X",
    "status": "completed",
    "dataset": "csl_daily",
    "split": "dev",
    "seeds": [42, 1337, 2026],
    "gain_gate_status": gain["status"],
    "mechanism_gate_status": mechanism["status"],
    "test_used_for_gate": False,
    "claim_scope": "cross_dataset_dev_screen_not_generality_or_test_claim",
}
atomic_json_dump(value, "artifacts/campaign/csl_gate_x_3seed.json")
print(json.dumps(value, sort_keys=True))
PY

printf '%s csl_multiseed_complete seeds=42,1337,2026 test_accessed=false\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
