#!/usr/bin/env bash
# Detached launch that survives agent-session teardown; re-running resumes from resume.pt.
# Usage: launch.sh RUN_DIR [train.py args...]
set -e
OUT=$1; shift
mkdir -p "$(dirname "$OUT")"
source /home/haipd/miniconda3/bin/activate seds
cd /home/haipd/SLR
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup python methods/sota_slret/src/train.py --out "$OUT" "$@" >> "$OUT.stdout" 2>&1 < /dev/null &
echo "launched pid $!"
