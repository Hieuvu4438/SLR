#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

log_path=artifacts/logs/ph_b512_local_word_video_multiseed.log
minimum_free_disk_gib=20
mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s local_word_video_multiseed_start pid=%s git_head=%s seeds=42,1337,2026\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s local_word_video_multiseed_exit pid=%s status=%s\n' \
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
  local teacher_run=$1
  local cache_path=$2
  local output=$3
  python -m elsc.configure_stage \
    --template configs/ablation_local_word_video.yaml \
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

run_gate() {
  local output=$1
  local pending_output="${output}.pending.$$"
  local cli_status=0
  python -m elsc.gate --gate G \
    --report artifacts/campaign/ph_local_word_video_vs_base_b512_dev_3seed.json \
    --output "$pending_output" >>"$log_path" 2>&1 || cli_status=$?
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

baseline_runs=()
local_runs=()

for seed in 42 1337 2026; do
  if [[ "$seed" == 42 ]]; then
    baseline_config=configs/ph_base.yaml
  else
    baseline_config="configs/ph_base_s${seed}.yaml"
  fi
  baseline_run="runs/ph_base_b512_s${seed}"
  cache_path="artifacts/cache/ph_min_b512_s${seed}_v1"
  local_config="artifacts/campaign/ph_local_word_video_b512_s${seed}.yaml"
  local_run="runs/ph_local_word_video_b512_s${seed}"

  baseline_runs+=("$baseline_run")
  local_runs+=("$local_run")
  python -m elsc.provenance --run-dir "$baseline_run" --config "$baseline_config" \
    >>"$log_path" 2>&1
  if [[ "$seed" != 42 ]]; then
    configure_control "$baseline_run" "$cache_path" "$local_config"
    run_training "$local_config" "$local_run" "local_word_video_s${seed}"
    check_disk
    python -m elsc.evaluate \
      --run-dir "$local_run" --split dev --checkpoint best_dev --device cuda:0 \
      >>"$log_path" 2>&1
    python -m elsc.report \
      --baseline-runs "$baseline_run" --method-runs "$local_run" --split dev \
      --output "artifacts/campaign/ph_local_word_video_vs_base_b512_dev_s${seed}.json" \
      >>"$log_path" 2>&1
  else
    python -m elsc.provenance --run-dir "$local_run" --config "$local_config" \
      >>"$log_path" 2>&1
  fi
done

python -m elsc.report \
  --baseline-runs "${baseline_runs[@]}" \
  --method-runs "${local_runs[@]}" \
  --split dev \
  --output artifacts/campaign/ph_local_word_video_vs_base_b512_dev_3seed.json \
  >>"$log_path" 2>&1
run_gate artifacts/campaign/ph_local_word_video_vs_base_b512_dev_3seed_gate_g.json

python - <<'PY' >>"$log_path" 2>&1
import json
from pathlib import Path

path = Path("artifacts/campaign/ph_local_word_video_vs_base_b512_dev_3seed_gate_g.json")
value = json.loads(path.read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "event": "local_word_video_multiseed_complete",
            "gate_g": value["status"],
            "test_accessed": False,
        },
        sort_keys=True,
    )
)
PY
printf '%s local_word_video_multiseed_complete test_accessed=false\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
