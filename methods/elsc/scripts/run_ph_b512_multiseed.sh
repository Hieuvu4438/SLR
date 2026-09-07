#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

log_path=artifacts/logs/ph_b512_multiseed.log
minimum_free_disk_gib=20
mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s multiseed_start pid=%s git_head=%s seeds=1337,2026\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s multiseed_exit pid=%s status=%s\n' \
    "$(date --iso-8601=seconds)" "$$" "$status" >>"$log_path"
}
trap campaign_exit EXIT

check_disk() {
  local available_bytes minimum_bytes
  available_bytes=$(df --output=avail -B1 /home/haipd/SLR | tail -n 1 | tr -d ' ')
  minimum_bytes=$((minimum_free_disk_gib * 1024 * 1024 * 1024))
  if (( available_bytes < minimum_bytes )); then
    printf '%s disk_guard_failed available_bytes=%s minimum_bytes=%s\n' \
      "$(date --iso-8601=seconds)" "$available_bytes" "$minimum_bytes" \
      >>"$log_path"
    return 1
  fi
  printf '%s disk_guard_passed available_bytes=%s minimum_bytes=%s\n' \
    "$(date --iso-8601=seconds)" "$available_bytes" "$minimum_bytes" \
    >>"$log_path"
}

run_training() {
  local config=$1
  local run_dir=$2
  local label=$3

  check_disk
  if python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
    >>"$log_path" 2>&1; then
    printf '%s run_already_complete_and_validated label=%s run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
    return 0
  fi
  if [[ -f "$run_dir/selection.json" || -f "$run_dir/run_summary.json" ]]; then
    printf '%s invalid_existing_run_refusing_overwrite label=%s run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
    return 1
  fi
  local args=(--config "$config" --run-dir "$run_dir" --device cuda:0)
  if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
    args+=(--resume "$run_dir/checkpoints/last.pt")
  fi
  printf '%s train_start label=%s run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m elsc.train "${args[@]}" >>"$log_path" 2>&1
  python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
    >>"$log_path" 2>&1
  printf '%s train_complete label=%s run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$label" "$run_dir" >>"$log_path"
}

evaluate_dev() {
  local run_dir=$1
  check_disk
  python -m elsc.evaluate \
    --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
    >>"$log_path" 2>&1
}

configure_stage() {
  local template=$1
  local teacher_run=$2
  local cache_path=$3
  local output=$4
  local args=(
    --template "$template"
    --teacher-run "$teacher_run"
    --output "$output"
  )
  if [[ -n "$cache_path" ]]; then
    args+=(--cache-path "$cache_path")
  fi
  python -m elsc.configure_stage "${args[@]}" >>"$log_path" 2>&1
}

report_pair() {
  local baseline_run=$1
  local method_run=$2
  local output=$3
  python -m elsc.report \
    --baseline-runs "$baseline_run" --method-runs "$method_run" --split dev \
    --output "$output" >>"$log_path" 2>&1
}

run_gate() {
  local expected_gate=$1
  local output=$2
  shift 2
  local pending_output="${output}.pending.$$"
  local cli_status=0

  "$@" --output "$pending_output" || cli_status=$?
  python - "$pending_output" "$expected_gate" "$cli_status" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_gate = sys.argv[2]
cli_status = int(sys.argv[3])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != 1 or value.get("gate") != expected_gate:
    raise ValueError(f"invalid Gate {expected_gate} artifact: {path}")
status = value.get("status")
if status not in {"passed", "no_go", "insufficient_seeds"}:
    raise ValueError(f"Gate {expected_gate} has invalid status: {status}")
if (status == "passed") != (cli_status == 0):
    raise ValueError(
        f"Gate {expected_gate} status/exit mismatch: status={status}, exit={cli_status}"
    )
print(json.dumps({"gate": expected_gate, "status": status, "exit": cli_status}))
PY
  mv -f -- "$pending_output" "$output"
}

run_seed() {
  local seed=$1
  local baseline_config="configs/ph_base_s${seed}.yaml"
  local baseline_run="runs/ph_base_b512_s${seed}"
  local cache_path="artifacts/cache/ph_min_b512_s${seed}_v1"
  local min_config="artifacts/campaign/ph_min_b512_s${seed}.yaml"
  local min_run="runs/ph_min_b512_s${seed}"

  run_training "$baseline_config" "$baseline_run" "baseline_s${seed}"
  evaluate_dev "$baseline_run"

  configure_stage methods/elsc/configs/ph_min.yaml "$baseline_run" "$cache_path" "$min_config"
  check_disk
  python -m elsc.mining.build_cache \
    --config "$min_config" --split train --device cuda:0 >>"$log_path" 2>&1
  run_training "$min_config" "$min_run" "min_s${seed}"
  evaluate_dev "$min_run"
  report_pair \
    "$baseline_run" "$min_run" \
    "artifacts/campaign/ph_min_vs_base_b512_dev_s${seed}.json"

  local control_name template control_config control_run
  for control_name in adapter_only caption random_span; do
    case "$control_name" in
      adapter_only) template=methods/elsc/configs/ablation_adapter_only.yaml ;;
      caption) template=methods/elsc/configs/ablation_caption.yaml ;;
      random_span) template=methods/elsc/configs/ablation_random_span.yaml ;;
    esac
    control_config="artifacts/campaign/ph_${control_name}_b512_s${seed}.yaml"
    control_run="runs/ph_${control_name}_b512_s${seed}"
    if [[ "$control_name" == adapter_only ]]; then
      configure_stage "$template" "$baseline_run" "" "$control_config"
    else
      configure_stage "$template" "$baseline_run" "$cache_path" "$control_config"
    fi
    run_training "$control_config" "$control_run" "${control_name}_s${seed}"
    evaluate_dev "$control_run"
    report_pair \
      "$baseline_run" "$control_run" \
      "artifacts/campaign/ph_${control_name}_vs_base_b512_dev_s${seed}.json"
    report_pair \
      "$control_run" "$min_run" \
      "artifacts/campaign/ph_min_vs_${control_name}_b512_dev_s${seed}.json"
  done
  printf '%s seed_complete seed=%s\n' \
    "$(date --iso-8601=seconds)" "$seed" >>"$log_path"
}

run_seed 1337
run_seed 2026

baseline_runs=(
  runs/ph_base_b512_s42
  runs/ph_base_b512_s1337
  runs/ph_base_b512_s2026
)
min_runs=(
  runs/ph_min_b512_s42
  runs/ph_min_b512_s1337
  runs/ph_min_b512_s2026
)
adapter_runs=(
  runs/ph_adapter_only_b512_s42
  runs/ph_adapter_only_b512_s1337
  runs/ph_adapter_only_b512_s2026
)
caption_runs=(
  runs/ph_caption_b512_s42
  runs/ph_caption_b512_s1337
  runs/ph_caption_b512_s2026
)
random_runs=(
  runs/ph_random_span_b512_s42
  runs/ph_random_span_b512_s1337
  runs/ph_random_span_b512_s2026
)

python -m elsc.report \
  --baseline-runs "${baseline_runs[@]}" --method-runs "${min_runs[@]}" --split dev \
  --output artifacts/campaign/ph_min_vs_base_b512_dev_3seed.json \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "${adapter_runs[@]}" --method-runs "${min_runs[@]}" --split dev \
  --output artifacts/campaign/ph_min_vs_adapter_only_b512_dev_3seed.json \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "${caption_runs[@]}" --method-runs "${min_runs[@]}" --split dev \
  --output artifacts/campaign/ph_min_vs_caption_b512_dev_3seed.json \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "${random_runs[@]}" --method-runs "${min_runs[@]}" --split dev \
  --output artifacts/campaign/ph_min_vs_random_span_b512_dev_3seed.json \
  >>"$log_path" 2>&1

run_gate G artifacts/campaign/ph_min_vs_base_b512_dev_3seed_gate_g.json \
  python -m elsc.gate --gate G \
  --report artifacts/campaign/ph_min_vs_base_b512_dev_3seed.json \
  >>"$log_path" 2>&1
run_gate M artifacts/campaign/ph_min_controls_b512_dev_3seed_gate_m.json \
  python -m elsc.gate --gate M \
  --true-vs-random-report \
    artifacts/campaign/ph_min_vs_random_span_b512_dev_3seed.json \
  --true-vs-caption-report \
    artifacts/campaign/ph_min_vs_caption_b512_dev_3seed.json \
  >>"$log_path" 2>&1

printf '%s multiseed_complete seeds=42,1337,2026\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
