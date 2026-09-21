#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

log_path=artifacts/logs/ph_uniformerv2_extraction.log
mkdir -p "$(dirname "$log_path")"

echo "========================================================" | tee -a "$log_path"
echo "Starting UniFormerV2 Feature Extraction for Phoenix14T" | tee -a "$log_path"
echo "Started at: $(date --iso-8601=seconds)" | tee -a "$log_path"
echo "========================================================" | tee -a "$log_path"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONUNBUFFERED=1 /home/haipd/miniconda3/envs/seds/bin/python -u scripts/extract_ph_uniformerv2.py \
    --splits test dev train \
    --stride 1 \
    --batch-size 8 \
    --device cuda:0 2>&1 | tee -a "$log_path"

echo "========================================================" | tee -a "$log_path"
echo "UniFormerV2 Feature Extraction Completed at: $(date --iso-8601=seconds)" | tee -a "$log_path"
echo "========================================================" | tee -a "$log_path"
