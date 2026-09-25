#!/usr/bin/env bash
# Restart the SLR job that was paused on 2026-09-23 to free the GPU for the
# CARVE ablation sweep. The training script auto-resumes from resume.pt, so this
# continues from the last completed epoch (19/50 at pause time).
cd /home/haipd/SLR/third_party/SEDS
nohup python methods/sota_slret/src/train.py \
  --out /home/haipd/SLR/runs/sota_slret/ph/E001_baseline_ep50_s42 \
  --dataset ph \
  --split_dir /home/haipd/SLR/artifacts/sota_slret_agent/splits/ph_val_s0 \
  --epochs 50 --seed 42 \
  >> /home/haipd/SLR/_paused/E001_resumed.log 2>&1 &
echo "restarted, pid $!  -- tail /home/haipd/SLR/_paused/E001_resumed.log"
