#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

log_path=artifacts/logs/ph_b512_lower_lr_multiseed.log
minimum_free_disk_gib=20
mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s lower_lr_multiseed_start pid=%s git_head=%s seeds=42,1337,2026\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s lower_lr_multiseed_exit pid=%s status=%s\n' \
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

configure_control() {
  local template=$1
  local teacher_run=$2
  local cache_path=$3
  local output=$4
  python -m elsc.configure_stage \
    --template "$template" \
    --teacher-run "$teacher_run" \
    --cache-path "$cache_path" \
    --output "$output" >>"$log_path" 2>&1
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

report_pair() {
  local baseline_run=$1
  local method_run=$2
  local output=$3
  local require_matched_budget=$4
  local budget_args=()
  if [[ "$require_matched_budget" == true ]]; then
    budget_args+=(--require-matched-training-budget)
  fi
  python -m elsc.report \
    --baseline-runs "$baseline_run" --method-runs "$method_run" --split dev \
    "${budget_args[@]}" --output "$output" >>"$log_path" 2>&1
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

gate_status() {
  local path=$1
  local expected=$2
  python - "$path" "$expected" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != 1 or value.get("gate") != expected:
    raise ValueError(f"invalid Gate {expected} artifact: {path}")
if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
    raise ValueError(f"Gate {expected} violates dev-only isolation")
print(value["status"])
PY
}

baseline_runs=()
lower_lr_runs=()
caption_runs=()
random_runs=()

for seed in 42 1337 2026; do
  if [[ "$seed" == 42 ]]; then
    baseline_config=configs/ph_base.yaml
  else
    baseline_config="configs/ph_base_s${seed}.yaml"
  fi
  baseline_run="runs/ph_base_b512_s${seed}"
  cache_path="artifacts/cache/ph_min_b512_s${seed}_v1"
  lower_lr_config="artifacts/campaign/ph_lower_lr_b512_s${seed}.yaml"
  lower_lr_run="runs/ph_lower_lr_b512_s${seed}"

  baseline_runs+=("$baseline_run")
  lower_lr_runs+=("$lower_lr_run")
  python -m elsc.provenance --run-dir "$baseline_run" --config "$baseline_config" \
    >>"$log_path" 2>&1
  if [[ "$seed" != 42 ]]; then
    configure_control configs/ablation_lower_lr.yaml \
      "$baseline_run" "$cache_path" "$lower_lr_config"
    run_training "$lower_lr_config" "$lower_lr_run" "lower_lr_s${seed}"
    evaluate_dev "$lower_lr_run"
    report_pair "$baseline_run" "$lower_lr_run" \
      "artifacts/campaign/ph_lower_lr_vs_base_b512_dev_s${seed}.json" false
  else
    python -m elsc.provenance --run-dir "$lower_lr_run" --config "$lower_lr_config" \
      >>"$log_path" 2>&1
  fi
done

python -m elsc.report \
  --baseline-runs "${baseline_runs[@]}" \
  --method-runs "${lower_lr_runs[@]}" \
  --split dev \
  --output artifacts/campaign/ph_lower_lr_vs_base_b512_dev_3seed.json \
  >>"$log_path" 2>&1
run_gate G artifacts/campaign/ph_lower_lr_vs_base_b512_dev_3seed_gate_g.json \
  python -m elsc.gate --gate G \
  --report artifacts/campaign/ph_lower_lr_vs_base_b512_dev_3seed.json \
  >>"$log_path" 2>&1

if [[ "$(gate_status artifacts/campaign/ph_lower_lr_vs_base_b512_dev_3seed_gate_g.json G)" \
  != passed ]]; then
  printf '%s lower_lr_multiseed_complete gate_g=no_go controls=skipped test_accessed=false\n' \
    "$(date --iso-8601=seconds)" >>"$log_path"
  exit 0
fi

for seed in 42 1337 2026; do
  baseline_run="runs/ph_base_b512_s${seed}"
  cache_path="artifacts/cache/ph_min_b512_s${seed}_v1"
  lower_lr_run="runs/ph_lower_lr_b512_s${seed}"
  caption_config="artifacts/campaign/ph_lower_lr_caption_b512_s${seed}.yaml"
  caption_run="runs/ph_lower_lr_caption_b512_s${seed}"
  random_config="artifacts/campaign/ph_lower_lr_random_span_b512_s${seed}.yaml"
  random_run="runs/ph_lower_lr_random_span_b512_s${seed}"

  configure_control configs/ablation_lower_lr_caption.yaml \
    "$baseline_run" "$cache_path" "$caption_config"
  run_training "$caption_config" "$caption_run" "lower_lr_caption_s${seed}"
  evaluate_dev "$caption_run"
  report_pair "$caption_run" "$lower_lr_run" \
    "artifacts/campaign/ph_lower_lr_true_vs_caption_b512_dev_s${seed}.json" true

  configure_control configs/ablation_lower_lr_random_span.yaml \
    "$baseline_run" "$cache_path" "$random_config"
  run_training "$random_config" "$random_run" "lower_lr_random_span_s${seed}"
  evaluate_dev "$random_run"
  report_pair "$random_run" "$lower_lr_run" \
    "artifacts/campaign/ph_lower_lr_true_vs_random_span_b512_dev_s${seed}.json" true

  caption_runs+=("$caption_run")
  random_runs+=("$random_run")
done

python -m elsc.report \
  --baseline-runs "${caption_runs[@]}" \
  --method-runs "${lower_lr_runs[@]}" \
  --split dev --require-matched-training-budget \
  --output artifacts/campaign/ph_lower_lr_true_vs_caption_b512_dev_3seed.json \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "${random_runs[@]}" \
  --method-runs "${lower_lr_runs[@]}" \
  --split dev --require-matched-training-budget \
  --output artifacts/campaign/ph_lower_lr_true_vs_random_span_b512_dev_3seed.json \
  >>"$log_path" 2>&1
run_gate M artifacts/campaign/ph_lower_lr_controls_b512_dev_3seed_gate_m.json \
  python -m elsc.gate --gate M \
  --true-vs-random-report \
    artifacts/campaign/ph_lower_lr_true_vs_random_span_b512_dev_3seed.json \
  --true-vs-caption-report \
    artifacts/campaign/ph_lower_lr_true_vs_caption_b512_dev_3seed.json \
  >>"$log_path" 2>&1

printf '%s lower_lr_multiseed_complete gate_g=passed gate_m=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" \
  "$(gate_status artifacts/campaign/ph_lower_lr_controls_b512_dev_3seed_gate_m.json M)" \
  >>"$log_path"
