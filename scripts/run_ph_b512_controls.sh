#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

baseline_run=runs/ph_base_b512_s42
baseline_config=configs/ph_base.yaml
min_run=runs/ph_min_b512_s42
min_config=artifacts/campaign/ph_min_b512_s42.yaml
cache_path=artifacts/cache/ph_min_b512_s42_v1
log_path=artifacts/logs/ph_b512_controls_s42.log

mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s controls_start pid=%s git_head=%s\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

controls_exit() {
  local status=$?
  printf '%s controls_exit pid=%s status=%s\n' \
    "$(date --iso-8601=seconds)" "$$" "$status" >>"$log_path"
}
trap controls_exit EXIT

python -m elsc.provenance --run-dir "$baseline_run" --config "$baseline_config" \
  >>"$log_path" 2>&1
python -m elsc.provenance --run-dir "$min_run" --config "$min_config" \
  >>"$log_path" 2>&1

configure_control() {
  local template=$1
  local output=$2
  local uses_cache=$3
  local args=(
    --template "$template"
    --teacher-run "$baseline_run"
    --output "$output"
  )
  if [[ "$uses_cache" == true ]]; then
    args+=(--cache-path "$cache_path")
  fi
  python -m elsc.configure_stage "${args[@]}" >>"$log_path" 2>&1
}

run_control() {
  local name=$1
  local template=$2
  local config=$3
  local run_dir=$4
  local uses_cache=$5

  configure_control "$template" "$config" "$uses_cache"
  if python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
    >>"$log_path" 2>&1; then
    printf '%s control_already_complete_and_validated name=%s run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$name" "$run_dir" >>"$log_path"
  else
    if [[ -f "$run_dir/selection.json" || -f "$run_dir/run_summary.json" ]]; then
      printf '%s invalid_existing_control_refusing_overwrite name=%s run_dir=%s\n' \
        "$(date --iso-8601=seconds)" "$name" "$run_dir" >>"$log_path"
      return 1
    fi
    local train_args=(
      --config "$config"
      --run-dir "$run_dir"
      --device cuda:0
    )
    if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
      train_args+=(--resume "$run_dir/checkpoints/last.pt")
    fi
    printf '%s control_train_start name=%s run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$name" "$run_dir" >>"$log_path"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -m elsc.train "${train_args[@]}" >>"$log_path" 2>&1
    python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
      >>"$log_path" 2>&1
  fi

  python -m elsc.evaluate \
    --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
    >>"$log_path" 2>&1
  python -m elsc.report \
    --baseline-runs "$baseline_run" --method-runs "$run_dir" --split dev \
    --output "artifacts/campaign/ph_${name}_vs_base_b512_dev_s42.json" \
    >>"$log_path" 2>&1
  python -m elsc.report \
    --baseline-runs "$run_dir" --method-runs "$min_run" --split dev \
    --output "artifacts/campaign/ph_min_vs_${name}_b512_dev_s42.json" \
    >>"$log_path" 2>&1
  printf '%s control_complete name=%s run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$name" "$run_dir" >>"$log_path"
}

run_control \
  adapter_only \
  configs/ablation_adapter_only.yaml \
  artifacts/campaign/ph_adapter_only_b512_s42.yaml \
  runs/ph_adapter_only_b512_s42 \
  false

run_control \
  caption \
  configs/ablation_caption.yaml \
  artifacts/campaign/ph_caption_b512_s42.yaml \
  runs/ph_caption_b512_s42 \
  true

run_control \
  random_span \
  configs/ablation_random_span.yaml \
  artifacts/campaign/ph_random_span_b512_s42.yaml \
  runs/ph_random_span_b512_s42 \
  true

printf '%s controls_seed42_complete\n' "$(date --iso-8601=seconds)" >>"$log_path"
