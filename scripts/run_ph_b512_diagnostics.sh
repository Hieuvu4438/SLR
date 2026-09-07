#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

gain_gate=artifacts/campaign/ph_min_vs_base_b512_dev_3seed_gate_g.json
mechanism_gate=artifacts/campaign/ph_min_controls_b512_dev_3seed_gate_m.json
baseline_run=runs/ph_base_b512_s42
baseline_config=configs/ph_base.yaml
legacy_min_run=runs/ph_min_b512_s42
legacy_min_config=artifacts/campaign/ph_min_b512_s42.yaml
matched_min_run=runs/ph_min_matched_source_b512_s42
matched_min_config=artifacts/campaign/ph_min_matched_source_b512_s42.yaml
min_cache=artifacts/cache/ph_min_b512_s42_v1
random_cache=artifacts/cache/ph_random_neighbors_b512_s42_v1
log_path=artifacts/logs/ph_b512_diagnostics_s42.log
minimum_free_disk_gib=20

mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s diagnostics_start pid=%s git_head=%s seed=42\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s diagnostics_exit pid=%s status=%s\n' \
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

require_terminal_gate() {
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
if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
    raise ValueError(f"Gate {expected} violates dev-only isolation")
allowed = {"passed", "no_go"} if expected == "G" else {
    "passed",
    "no_go",
    "insufficient_seeds",
}
if value.get("status") not in allowed:
    raise ValueError(f"Gate {expected} is not terminal: {value.get('status')}")
print(json.dumps({"gate": expected, "status": value["status"], "path": str(path.resolve())}))
PY
}

configure_control() {
  local template=$1
  local cache_path=$2
  local output=$3
  python -m elsc.configure_stage \
    --template "$template" \
    --teacher-run "$baseline_run" \
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
  local name=$1
  local run_dir=$2
  local require_matched_budget=$3

  check_disk
  python -m elsc.evaluate \
    --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
    >>"$log_path" 2>&1
  python -m elsc.report \
    --baseline-runs "$baseline_run" --method-runs "$run_dir" --split dev \
    --output "artifacts/campaign/ph_${name}_vs_base_b512_dev_s42.json" \
    >>"$log_path" 2>&1
  local budget_args=()
  if [[ "$require_matched_budget" == true ]]; then
    budget_args+=(--require-matched-training-budget)
  fi
  python -m elsc.report \
    --baseline-runs "$run_dir" --method-runs "$matched_min_run" --split dev \
    --output "artifacts/campaign/ph_min_vs_${name}_b512_dev_s42.json" \
    "${budget_args[@]}" >>"$log_path" 2>&1
}

check_disk
require_terminal_gate "$gain_gate" G
require_terminal_gate "$mechanism_gate" M
python -m elsc.provenance --run-dir "$baseline_run" --config "$baseline_config" \
  >>"$log_path" 2>&1
python -m elsc.provenance --run-dir "$legacy_min_run" --config "$legacy_min_config" \
  >>"$log_path" 2>&1

# The registered Min run predates the Full/KEEP source refactor. Although those
# branches are disabled for Min, the fail-closed report contract correctly
# rejects comparisons across different implementation fingerprints. Reproduce
# Min under the current source tree instead of weakening the contract or
# overwriting the registered run.
configure_control configs/ph_min.yaml "$min_cache" "$matched_min_config"
run_training "$matched_min_config" "$matched_min_run" min_matched_source
check_disk
python -m elsc.evaluate \
  --run-dir "$matched_min_run" --split dev --checkpoint best_dev --device cuda:0 \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "$legacy_min_run" --method-runs "$matched_min_run" --split dev \
  --output artifacts/campaign/ph_min_matched_source_vs_legacy_b512_dev_s42.json \
  >>"$log_path" 2>&1

for name in shuffled_lexical head_only local_word_video; do
  case "$name" in
    shuffled_lexical) template=configs/ablation_shuffled_lexical.yaml ;;
    head_only) template=configs/ablation_head_only.yaml ;;
    local_word_video) template=configs/ablation_local_word_video.yaml ;;
  esac
  config="artifacts/campaign/ph_${name}_b512_s42.yaml"
  run_dir="runs/ph_${name}_b512_s42"
  configure_control "$template" "$min_cache" "$config"
  run_training "$config" "$run_dir" "$name"
  if [[ "$name" == head_only ]]; then
    evaluate_and_report "$name" "$run_dir" false
  else
    evaluate_and_report "$name" "$run_dir" true
  fi
done

random_config=artifacts/campaign/ph_random_neighbors_b512_s42.yaml
random_run=runs/ph_random_neighbors_b512_s42
configure_control configs/ablation_random_neighbors.yaml "$random_cache" "$random_config"
check_disk
python -m elsc.mining.build_cache \
  --config "$random_config" --split train --device cuda:0 >>"$log_path" 2>&1
run_training "$random_config" "$random_run" random_neighbors
evaluate_and_report random_neighbors "$random_run" true

printf '%s diagnostics_complete seed=42 test_accessed=false\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
