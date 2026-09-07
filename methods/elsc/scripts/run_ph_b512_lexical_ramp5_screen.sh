#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

baseline_run=runs/ph_base_b512_s42
config=artifacts/campaign/ph_lexical_ramp5_b512_s42.yaml
run_dir=runs/ph_lexical_ramp5_b512_s42
report=artifacts/campaign/ph_lexical_ramp5_vs_base_b512_dev_s42.json
gate=artifacts/campaign/ph_lexical_ramp5_vs_base_b512_dev_s42_gate_g.json
log_path=artifacts/logs/ph_b512_lexical_ramp5_screen.log
minimum_free_disk_gib=20

mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s lexical_ramp5_start pid=%s git_head=%s seed=42\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

campaign_exit() {
  local status=$?
  printf '%s lexical_ramp5_exit pid=%s status=%s\n' \
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

check_disk
python -m elsc.provenance --run-dir "$baseline_run" --config configs/ph_base.yaml \
  >>"$log_path" 2>&1
python -m elsc.configure_stage \
  --template methods/elsc/configs/ablation_lexical_ramp5.yaml \
  --teacher-run "$baseline_run" \
  --cache-path artifacts/cache/ph_min_b512_s42_v1 \
  --output "$config" >>"$log_path" 2>&1

if python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
  >>"$log_path" 2>&1; then
  printf '%s run_already_complete_and_validated run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$run_dir" >>"$log_path"
else
  if [[ -f "$run_dir/selection.json" || -f "$run_dir/run_summary.json" ]]; then
    printf '%s invalid_existing_run_refusing_overwrite run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$run_dir" >>"$log_path"
    exit 1
  fi
  args=(--config "$config" --run-dir "$run_dir" --device cuda:0)
  if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
    args+=(--resume "$run_dir/checkpoints/last.pt")
  fi
  printf '%s train_start run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$run_dir" >>"$log_path"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m elsc.train "${args[@]}" >>"$log_path" 2>&1
  python -m elsc.provenance --run-dir "$run_dir" --config "$config" \
    >>"$log_path" 2>&1
  printf '%s train_complete run_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$run_dir" >>"$log_path"
fi

check_disk
python -m elsc.evaluate \
  --run-dir "$run_dir" --split dev --checkpoint best_dev --device cuda:0 \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "$baseline_run" --method-runs "$run_dir" --split dev \
  --output "$report" >>"$log_path" 2>&1

pending_gate="${gate}.pending.$$"
gate_exit=0
python -m elsc.gate --gate G --report "$report" --output "$pending_gate" \
  >>"$log_path" 2>&1 || gate_exit=$?
python - "$pending_gate" "$gate_exit" >>"$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
gate_exit = int(sys.argv[2])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != 1 or value.get("gate") != "G":
    raise ValueError(f"invalid Gate G artifact: {path}")
if value.get("split") != "dev" or value.get("test_used_for_gate") is not False:
    raise ValueError("Gate G violates dev-only isolation")
status = value.get("status")
if status not in {"passed", "no_go", "insufficient_seeds"}:
    raise ValueError(f"invalid Gate G status: {status}")
if (status == "passed") != (gate_exit == 0):
    raise ValueError(f"Gate G status/exit mismatch: status={status}, exit={gate_exit}")
print(json.dumps({"gate": "G", "status": status, "exit": gate_exit}))
PY
mv -f -- "$pending_gate" "$gate"

python - "$gate" >>"$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "event": "lexical_ramp5_complete",
            "gate_g": value["status"],
            "seed": 42,
            "test_accessed": False,
        },
        sort_keys=True,
    )
)
PY
printf '%s lexical_ramp5_complete seed=42 test_accessed=false\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
