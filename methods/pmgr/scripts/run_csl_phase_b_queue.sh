#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR
export PYTHONPATH=methods/pmgr/src:shared

base_config=methods/pmgr/configs/pmgr_csl.json
log_path=artifacts/pmgr/logs/csl_phase_b_seed0_queue.log
report_path=artifacts/pmgr/csl_phase_b_seed0_report.json
diagnostics_dir=artifacts/pmgr/diagnostics/csl_phase_b_seed0
wait_timeout_seconds=${PMGR_GPU_WAIT_TIMEOUT_SECONDS:-1209600}
poll_seconds=${PMGR_GPU_POLL_SECONDS:-60}
minimum_free_gpu_mib=${PMGR_MIN_FREE_GPU_MIB:-42000}
maximum_gpu_utilization=${PMGR_MAX_GPU_UTILIZATION:-10}
minimum_free_disk_gib=${PMGR_MIN_FREE_DISK_GIB:-20}
stable_polls=${PMGR_GPU_STABLE_POLLS:-3}
run_timeout_seconds=${PMGR_RUN_TIMEOUT_SECONDS:-604800}
expected_git_head=${PMGR_EXPECTED_GIT_HEAD:-$(git rev-parse HEAD)}

mkdir -p "$(dirname "$log_path")" "$diagnostics_dir"
exec 9>/tmp/slr_pmgr_gpu0.lock
if ! flock -n 9; then
  printf '%s queue_refused reason=pmgr_lock_held test_accessed=false\n' \
    "$(date --iso-8601=seconds)" >>"$log_path"
  exit 3
fi

printf '%s queue_start pid=%s git_head=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$$" "$expected_git_head" >>"$log_path"

queue_exit() {
  local status=$?
  printf '%s queue_exit pid=%s status=%s test_accessed=false\n' \
    "$(date --iso-8601=seconds)" "$$" "$status" >>"$log_path"
}
trap queue_exit EXIT

check_code_state() {
  local actual_git_head
  actual_git_head=$(git rev-parse HEAD)
  if [[ "$actual_git_head" != "$expected_git_head" ]]; then
    printf '%s code_guard_failed reason=head_changed expected=%s actual=%s\n' \
      "$(date --iso-8601=seconds)" "$expected_git_head" "$actual_git_head" >>"$log_path"
    return 1
  fi
  if [[ -n "$(git status --porcelain -- conftest.py methods/pmgr methods/README.md pyproject.toml)" ]]; then
    printf '%s code_guard_failed reason=uncommitted_pmgr_change\n' \
      "$(date --iso-8601=seconds)" >>"$log_path"
    return 1
  fi
}

check_disk() {
  local available_bytes minimum_bytes
  available_bytes=$(df --output=avail -B1 /home/haipd/SLR | tail -n 1 | tr -d ' ')
  minimum_bytes=$((minimum_free_disk_gib * 1024 * 1024 * 1024))
  if (( available_bytes < minimum_bytes )); then
    printf '%s disk_guard_failed available_bytes=%s minimum_bytes=%s\n' \
      "$(date --iso-8601=seconds)" "$available_bytes" "$minimum_bytes" >>"$log_path"
    return 1
  fi
}

wait_for_gpu() {
  local started_at now elapsed free_mib utilization stable=0
  started_at=$(date +%s)
  while true; do
    IFS=, read -r free_mib utilization < <(
      nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' '
    )
    if (( free_mib >= minimum_free_gpu_mib && utilization <= maximum_gpu_utilization )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    printf '%s gpu_wait free_mib=%s utilization=%s stable=%s/%s\n' \
      "$(date --iso-8601=seconds)" "$free_mib" "$utilization" "$stable" "$stable_polls" \
      >>"$log_path"
    if (( stable >= stable_polls )); then
      return 0
    fi
    now=$(date +%s)
    elapsed=$((now - started_at))
    if (( elapsed >= wait_timeout_seconds )); then
      printf '%s gpu_wait_timeout elapsed=%s limit=%s\n' \
        "$(date --iso-8601=seconds)" "$elapsed" "$wait_timeout_seconds" >>"$log_path"
      return 124
    fi
    sleep "$poll_seconds"
  done
}

validate_run() {
  local run_dir=$1 expected_status=$2
  python - "$run_dir" "$expected_status" <<'PY' >>"$log_path" 2>&1
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected = sys.argv[2]
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
if summary.get("status") != expected:
    raise ValueError(f"{run}: {summary.get('status')} != {expected}")
if config["paths"].get("test_index") is not None or config["paths"].get("test_manifest") is not None:
    raise ValueError("pilot configuration unexpectedly unlocks test")
if summary.get("checkpoint_reload_score_max_abs") != 0.0:
    raise ValueError("checkpoint reload score parity failed")
print(json.dumps({"run": str(run), "status": expected, "test_accessed": False}))
PY
}

common_args=(
  --config "$base_config"
  --engine two_level_replay
  --effective-groups 512
  --video-encoder-microbatch 16
  --text-encoder-microbatch 64
  --score-video-block 32
  --score-text-block 64
  --device cuda:0
)

check_code_state
check_disk
python -m pmgr.runtime audit --config "$base_config" \
  --output artifacts/pmgr/phase_b_launch_audit.json >>"$log_path" 2>&1

preflight_run=runs/pmgr_csl_phase_b_preflight_g512
if ! validate_run "$preflight_run" max_updates_reached; then
  wait_for_gpu
  check_code_state
  check_disk
  preflight_args=("${common_args[@]}" --loss-mode group_ce --max-updates 1 --output-dir "$preflight_run")
  if [[ -f "$preflight_run/checkpoints/last.pt" ]]; then
    preflight_args+=(--resume "$preflight_run/checkpoints/last.pt")
  fi
  printf '%s preflight_start effective_groups=512\n' "$(date --iso-8601=seconds)" >>"$log_path"
  timeout "$run_timeout_seconds" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m pmgr.train "${preflight_args[@]}" >>"$log_path" 2>&1
  validate_run "$preflight_run" max_updates_reached
fi

run_arm() {
  local arm=$1 mode=$2 run_dir=$3
  if validate_run "$run_dir" training_complete; then
    printf '%s arm_skip_complete arm=%s run=%s\n' \
      "$(date --iso-8601=seconds)" "$arm" "$run_dir" >>"$log_path"
    return 0
  fi
  wait_for_gpu
  check_code_state
  check_disk
  local args=("${common_args[@]}" --loss-mode "$mode" --output-dir "$run_dir")
  if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
    args+=(--resume "$run_dir/checkpoints/last.pt")
  fi
  printf '%s arm_start arm=%s mode=%s run=%s\n' \
    "$(date --iso-8601=seconds)" "$arm" "$mode" "$run_dir" >>"$log_path"
  timeout "$run_timeout_seconds" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m pmgr.train "${args[@]}" >>"$log_path" 2>&1
  validate_run "$run_dir" training_complete
  printf '%s arm_complete arm=%s mode=%s run=%s\n' \
    "$(date --iso-8601=seconds)" "$arm" "$mode" "$run_dir" >>"$log_path"
}

run_arm C0 legacy_cico runs/pmgr_csl_c0_legacy_seed0
run_arm C1 single_mixed_ce runs/pmgr_csl_c1_single_mixed_seed0
run_arm C2 all_uniform_ce runs/pmgr_csl_c2_uniform_seed0
run_arm C3 all_set_ce runs/pmgr_csl_c3_set_seed0
run_arm C3_population all_set_ce_population_weighted runs/pmgr_csl_c3_set_population_seed0
run_arm C4 group_ce runs/pmgr_csl_c4_group_seed0

python -m pmgr.report \
  --c0 runs/pmgr_csl_c0_legacy_seed0 \
  --c1 runs/pmgr_csl_c1_single_mixed_seed0 \
  --c2 runs/pmgr_csl_c2_uniform_seed0 \
  --c3 runs/pmgr_csl_c3_set_seed0 \
  --c3-population runs/pmgr_csl_c3_set_population_seed0 \
  --c4 runs/pmgr_csl_c4_group_seed0 \
  --output "$report_path" >>"$log_path" 2>&1

python -m pmgr.diagnostics --config "$base_config" --mode population \
  --checkpoint runs/pmgr_csl_c4_group_seed0/checkpoints/best_dev.pt \
  --output-dir "$diagnostics_dir" --groups 8 --draws 5 --device cuda:0 \
  >>"$log_path" 2>&1
python -m pmgr.diagnostics --config "$base_config" --mode weak_performance \
  --baseline-checkpoint runs/csl_base_b512_s42/checkpoints/best_dev.pt \
  --candidate-checkpoint runs/pmgr_csl_c4_group_seed0/checkpoints/best_dev.pt \
  --output-dir "$diagnostics_dir" --device cuda:0 >>"$log_path" 2>&1

printf '%s phase_b_complete report=%s test_accessed=false\n' \
  "$(date --iso-8601=seconds)" "$report_path" >>"$log_path"
