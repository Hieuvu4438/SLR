#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

log_path=artifacts/logs/ph_b512_balanced_lexical.log
minimum_free_disk_gib=20
mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s balanced_lexical_start pid=%s git_head=%s seeds=42,1337,2026\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s balanced_lexical_exit pid=%s status=%s\n' \
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

configure_run() {
  local teacher_run=$1
  local cache_path=$2
  local output=$3
  python -m elsc.configure_stage \
    --template methods/elsc/configs/ablation_balanced_lexical.yaml \
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

evaluate_and_report() {
  local baseline_run=$1
  local run_dir=$2
  local output=$3
  check_disk
  python -m elsc.evaluate \
    --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
    >>"$log_path" 2>&1
  python -m elsc.report \
    --baseline-runs "$baseline_run" --method-runs "$run_dir" --split dev \
    --output "$output" >>"$log_path" 2>&1
}

run_gate() {
  local report=$1
  local output=$2
  local pending_output="${output}.pending.$$"
  local cli_status=0
  python -m elsc.gate --gate G --report "$report" --output "$pending_output" \
    >>"$log_path" 2>&1 || cli_status=$?
  python - "$pending_output" "$cli_status" >>"$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
cli_status = int(sys.argv[2])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != 1 or value.get("gate") != "G":
    raise ValueError(f"invalid Gate G artifact: {path}")
status = value.get("status")
if status not in {"passed", "no_go", "insufficient_seeds"}:
    raise ValueError(f"Gate G has invalid status: {status}")
if (status == "passed") != (cli_status == 0):
    raise ValueError(f"Gate G status/exit mismatch: status={status}, exit={cli_status}")
print(json.dumps({"gate": "G", "status": status, "exit": cli_status}))
PY
  mv -f -- "$pending_output" "$output"
}

gate_status() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_version") != 1 or value.get("gate") != "G":
    raise ValueError("invalid Gate G artifact")
if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
    raise ValueError("Gate G violates dev-only isolation")
print(value["status"])
PY
}

baseline_runs=()
balanced_runs=()

for seed in 42 1337 2026; do
  if [[ "$seed" == 42 ]]; then
    baseline_config=configs/ph_base.yaml
  else
    baseline_config="configs/ph_base_s${seed}.yaml"
  fi
  baseline_run="runs/ph_base_b512_s${seed}"
  cache_path="artifacts/cache/ph_min_b512_s${seed}_v1"
  balanced_config="artifacts/campaign/ph_balanced_lexical_b512_s${seed}.yaml"
  balanced_run="runs/ph_balanced_lexical_b512_s${seed}"

  python -m elsc.provenance --run-dir "$baseline_run" --config "$baseline_config" \
    >>"$log_path" 2>&1
  configure_run "$baseline_run" "$cache_path" "$balanced_config"
  run_training "$balanced_config" "$balanced_run" "balanced_lexical_s${seed}"
  evaluate_and_report "$baseline_run" "$balanced_run" \
    "artifacts/campaign/ph_balanced_lexical_vs_base_b512_dev_s${seed}.json"

  baseline_runs+=("$baseline_run")
  balanced_runs+=("$balanced_run")

  if [[ "$seed" == 42 ]]; then
    seed42_gate=artifacts/campaign/ph_balanced_lexical_vs_base_b512_dev_s42_gate_g.json
    run_gate artifacts/campaign/ph_balanced_lexical_vs_base_b512_dev_s42.json \
      "$seed42_gate"
    if [[ "$(gate_status "$seed42_gate")" != passed ]]; then
      printf '%s balanced_lexical_complete seed42_gate_g=no_go expansion=skipped test_accessed=false\n' \
        "$(date --iso-8601=seconds)" >>"$log_path"
      exit 0
    fi
  fi
done

report=artifacts/campaign/ph_balanced_lexical_vs_base_b512_dev_3seed.json
gate=artifacts/campaign/ph_balanced_lexical_vs_base_b512_dev_3seed_gate_g.json
python -m elsc.report \
  --baseline-runs "${baseline_runs[@]}" \
  --method-runs "${balanced_runs[@]}" \
  --split dev --output "$report" >>"$log_path" 2>&1
run_gate "$report" "$gate"
printf '%s balanced_lexical_complete three_seed_gate_g=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$(gate_status "$gate")" >>"$log_path"
