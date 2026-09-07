#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

baseline_config=configs/ph_base.yaml
baseline_run=runs/ph_base_b512_s42
min_run=runs/ph_min_b512_s42
min_config=artifacts/campaign/ph_min_b512_s42.yaml
min_cache=artifacts/cache/ph_min_b512_s42_v1
report=artifacts/campaign/ph_min_vs_base_b512_dev_s42.json
log_path=artifacts/logs/ph_b512_followup_s42.log
wait_timeout_seconds=${BASELINE_WAIT_TIMEOUT_SECONDS:-5400}
poll_seconds=${BASELINE_POLL_SECONDS:-60}

mkdir -p artifacts/campaign "$(dirname "$log_path")"
printf '%s followup_start pid=%s git_head=%s\n' \
  "$(date --iso-8601=seconds)" "$$" "$(git rev-parse HEAD)" >>"$log_path"

followup_exit() {
  local status=$?
  printf '%s followup_exit pid=%s status=%s\n' \
    "$(date --iso-8601=seconds)" "$$" "$status" >>"$log_path"
}
trap followup_exit EXIT

started_at=$(date +%s)
while [[ ! -f "$baseline_run/run_summary.json" || ! -f "$baseline_run/selection.json" ]]; do
  now=$(date +%s)
  elapsed=$((now - started_at))
  if ! pgrep -f -- "python -m elsc.train --config $baseline_config --run-dir $baseline_run" \
    >/dev/null; then
    printf '%s baseline_process_missing elapsed=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
    exit 1
  fi
  if (( elapsed >= wait_timeout_seconds )); then
    printf '%s baseline_wait_timeout elapsed=%s limit=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" "$wait_timeout_seconds" >>"$log_path"
    exit 124
  fi
  printf '%s waiting_for_baseline elapsed=%s\n' \
    "$(date --iso-8601=seconds)" "$elapsed" >>"$log_path"
  sleep "$poll_seconds"
done

# The summary is the last durable artifact, but wait for the trainer to release
# its CUDA context before launching evaluation/mining on the same device.
while pgrep -f -- "python -m elsc.train --config $baseline_config --run-dir $baseline_run" \
  >/dev/null; do
  now=$(date +%s)
  elapsed=$((now - started_at))
  if (( elapsed >= wait_timeout_seconds )); then
    printf '%s baseline_exit_timeout elapsed=%s limit=%s\n' \
      "$(date --iso-8601=seconds)" "$elapsed" "$wait_timeout_seconds" >>"$log_path"
    exit 124
  fi
  sleep 2
done

python - "$baseline_config" "$baseline_run" >>"$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

from elsc.config import config_hash, load_config
from elsc.provenance import validate_dev_selection

config_path = Path(sys.argv[1])
run = Path(sys.argv[2])
selection_path = run / "selection.json"
checkpoint = run / "checkpoints" / "best_dev.pt"
selection = validate_dev_selection(selection_path, checkpoint)
summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
expected_hash = config_hash(load_config(config_path, stage="train"))
if summary.get("status") != "complete":
    raise ValueError("baseline run summary is not complete")
if selection.get("config_hash") != expected_hash:
    raise ValueError("baseline selection was produced by a different config")
print(json.dumps({"baseline_validation": "passed", "config_hash": expected_hash}))
PY

python -m elsc.evaluate \
  --run-dir "$baseline_run" --split dev --checkpoint best_dev --device cuda:0 \
  >>"$log_path" 2>&1

python -m elsc.configure_stage \
  --template methods/elsc/configs/ph_min.yaml \
  --teacher-run "$baseline_run" \
  --cache-path "$min_cache" \
  --output "$min_config" >>"$log_path" 2>&1

python -m elsc.mining.build_cache \
  --config "$min_config" --split train --device cuda:0 >>"$log_path" 2>&1

if [[ -f "$min_run/checkpoints/last.pt" && ! -f "$min_run/selection.json" ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m elsc.train \
    --config "$min_config" --run-dir "$min_run" \
    --resume "$min_run/checkpoints/last.pt" --device cuda:0 >>"$log_path" 2>&1
elif [[ ! -f "$min_run/selection.json" ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m elsc.train \
    --config "$min_config" --run-dir "$min_run" --device cuda:0 >>"$log_path" 2>&1
fi

python - "$min_config" "$min_run" >>"$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

from elsc.config import config_hash, load_config
from elsc.provenance import validate_dev_selection

config_path = Path(sys.argv[1])
run = Path(sys.argv[2])
selection = validate_dev_selection(
    run / "selection.json", run / "checkpoints" / "best_dev.pt"
)
summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
expected_hash = config_hash(load_config(config_path, stage="train"))
if summary.get("status") != "complete":
    raise ValueError("ELSC-Min run summary is not complete")
if selection.get("config_hash") != expected_hash:
    raise ValueError("ELSC-Min selection was produced by a different config")
print(json.dumps({"min_validation": "passed", "config_hash": expected_hash}))
PY

python -m elsc.evaluate \
  --run-dir "$min_run" --split dev --checkpoint best_dev --device cuda:0 \
  >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs "$baseline_run" --method-runs "$min_run" \
  --split dev --output "$report" >>"$log_path" 2>&1

printf '%s b512_seed42_dev_screen_complete\n' \
  "$(date --iso-8601=seconds)" >>"$log_path"
