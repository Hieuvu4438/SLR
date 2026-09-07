#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

gain_gate=artifacts/campaign/ph_min_vs_base_b512_dev_3seed_gate_g.json
baseline_run=runs/ph_base_b512_s42
baseline_config=configs/ph_base.yaml
min_run=runs/ph_min_b512_s42
min_config=artifacts/campaign/ph_min_b512_s42.yaml
min_cache=artifacts/cache/ph_min_b512_s42_v1
keep_config=artifacts/campaign/ph_keep_b512_s42.yaml
lower_lr_config=artifacts/campaign/ph_lower_lr_b512_s42.yaml
keep_run=runs/ph_keep_b512_s42
lower_lr_run=runs/ph_lower_lr_b512_s42
keep_preflight=artifacts/preflight/ph_keep_b512_s42.json
log_path=artifacts/logs/ph_b512_corrective_s42.log
minimum_free_disk_gib=20

mkdir -p artifacts/campaign artifacts/preflight "$(dirname "$log_path")"
printf '%s corrective_start pid=%s git_head=%s seed=42\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s corrective_exit pid=%s status=%s\n' \
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

gate_status=$(python - "$gain_gate" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != 1 or value.get("gate") != "G":
    raise ValueError(f"invalid Gate G artifact: {path}")
if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
    raise ValueError("Gate G violates dev-only isolation")
if value.get("status") not in {"passed", "no_go"}:
    raise ValueError(f"Gate G is not terminal: {value.get('status')}")
print(value["status"])
PY
)
if [[ "$gate_status" == passed ]]; then
  printf '%s corrective_not_applicable gate_g=passed\n' \
    "$(date --iso-8601=seconds)" >>"$log_path"
  exit 0
fi

configure_control() {
  local template=$1
  local output=$2
  python -m elsc.configure_stage \
    --template "$template" \
    --teacher-run "$baseline_run" \
    --cache-path "$min_cache" \
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
  check_disk
  python -m elsc.evaluate \
    --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
    >>"$log_path" 2>&1
  python -m elsc.report \
    --baseline-runs "$baseline_run" --method-runs "$run_dir" --split dev \
    --output "artifacts/campaign/ph_${name}_vs_base_b512_dev_s42.json" \
    >>"$log_path" 2>&1
  python -m elsc.report \
    --baseline-runs "$min_run" --method-runs "$run_dir" --split dev \
    --output "artifacts/campaign/ph_${name}_vs_min_b512_dev_s42.json" \
    >>"$log_path" 2>&1
}

check_disk
python -m elsc.provenance --run-dir "$baseline_run" --config "$baseline_config" \
  >>"$log_path" 2>&1
python -m elsc.provenance --run-dir "$min_run" --config "$min_config" \
  >>"$log_path" 2>&1

configure_control methods/elsc/configs/ablation_lower_lr.yaml "$lower_lr_config"
run_training "$lower_lr_config" "$lower_lr_run" lower_lr
evaluate_and_report lower_lr "$lower_lr_run"

configure_control methods/elsc/configs/ablation_keep.yaml "$keep_config"
check_disk
python -m elsc.training_preflight \
  --config "$keep_config" \
  --batch-size 512 \
  --min-free-after-gib 4 \
  --output "$keep_preflight" \
  --device cuda:0 >>"$log_path" 2>&1
run_training "$keep_config" "$keep_run" keep_005
evaluate_and_report keep "$keep_run"

python -m elsc.report \
  --baseline-runs "$lower_lr_run" --method-runs "$keep_run" --split dev \
  --output artifacts/campaign/ph_keep_vs_lower_lr_b512_dev_s42.json \
  >>"$log_path" 2>&1

printf '%s corrective_complete seed=42 test_accessed=false\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
