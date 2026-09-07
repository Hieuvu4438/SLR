#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

gain_gate=artifacts/campaign/ph_min_vs_base_b512_dev_3seed_gate_g.json
mechanism_gate=artifacts/campaign/ph_min_controls_b512_dev_3seed_gate_m.json
baseline_run=runs/ph_base_b512_s42
min_run=runs/ph_min_b512_s42
min_cache=artifacts/cache/ph_min_b512_s42_v1
full_cache=artifacts/cache/ph_full_b512_s42_v1
full_config=artifacts/campaign/ph_full_b512_s42.yaml
continued_config=artifacts/campaign/ph_continued_min_b512_s42.yaml
full_run=runs/ph_full_b512_s42
continued_run=runs/ph_continued_min_b512_s42
full_gate=artifacts/campaign/ph_full_b512_s42_gate_f.json
preflight=artifacts/preflight/ph_full_b512_s42.json
log_path=artifacts/logs/ph_b512_full_pilot_s42.log
minimum_free_disk_gib=20

mkdir -p artifacts/campaign artifacts/preflight "$(dirname "$log_path")"
printf '%s full_pilot_start pid=%s git_head=%s\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s full_pilot_exit pid=%s status=%s\n' \
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

require_passed_gate() {
  local path=$1
  local expected=$2
  python - "$path" "$expected" >>"$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != 1 or value.get("gate") != expected:
    raise ValueError(f"invalid Gate {expected} artifact: {path}")
if value.get("status") != "passed":
    raise ValueError(f"Gate {expected} did not pass: {value.get('status')}")
if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
    raise ValueError(f"Gate {expected} violates dev-only isolation")
print(json.dumps({"gate": expected, "status": "passed", "path": str(path.resolve())}))
PY
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
  local baseline=$1
  local method=$2
  local output=$3
  shift 3
  python -m elsc.report \
    --baseline-runs "$baseline" --method-runs "$method" --split dev \
    --output "$output" "$@" >>"$log_path" 2>&1
}

check_disk
require_passed_gate "$gain_gate" G
require_passed_gate "$mechanism_gate" M
python -m elsc.provenance --run-dir "$baseline_run" --config configs/ph_base.yaml \
  >>"$log_path" 2>&1
python -m elsc.provenance --run-dir "$min_run" \
  --config artifacts/campaign/ph_min_b512_s42.yaml >>"$log_path" 2>&1

python -m elsc.configure_stage \
  --template configs/ph_full.yaml \
  --teacher-run "$baseline_run" \
  --student-run "$min_run" \
  --cache-path "$full_cache" \
  --output "$full_config" >>"$log_path" 2>&1
python -m elsc.configure_stage \
  --template configs/ablation_continued_min.yaml \
  --teacher-run "$baseline_run" \
  --student-run "$min_run" \
  --cache-path "$min_cache" \
  --output "$continued_config" >>"$log_path" 2>&1
python - "$continued_config" "$full_config" >>"$log_path" 2>&1 <<'PY'
import json
import sys

from elsc.config import load_config
from elsc.report import configured_optimization_budget
from elsc.utils import sha256_json

reference = configured_optimization_budget(load_config(sys.argv[1], stage="train"))
candidate = configured_optimization_budget(load_config(sys.argv[2], stage="train"))
if reference != candidate:
    raise ValueError("Full and continued-Min optimization budgets differ")
print(json.dumps({"optimization_budget": "matched", "sha256": sha256_json(reference)}))
PY
check_disk
python -m elsc.mining.build_cache \
  --config "$full_config" --split train --device cuda:0 >>"$log_path" 2>&1

python -m elsc.gate --gate F \
  --gain-gate "$gain_gate" \
  --mechanism-gate "$mechanism_gate" \
  --config "$full_config" \
  --cache "$full_cache" \
  --output "$full_gate" >>"$log_path" 2>&1
require_passed_gate "$full_gate" F

check_disk
python -m elsc.training_preflight \
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
  artifacts/campaign/ph_full_vs_continued_min_b512_dev_s42.json \
  --require-matched-training-budget
report_pair \
  "$min_run" "$full_run" \
  artifacts/campaign/ph_full_vs_min_b512_dev_s42.json
report_pair \
  "$baseline_run" "$full_run" \
  artifacts/campaign/ph_full_vs_base_b512_dev_s42.json

printf '%s full_pilot_complete seed=42 test_accessed=false\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
