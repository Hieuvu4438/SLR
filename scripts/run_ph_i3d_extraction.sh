#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

video_root=/home/dongvk/datasets/phoenix14T/videos_phoenix/videos
temporal_root=artifacts/sign_features/ph_temporal_metadata
log_path=artifacts/logs/ph_i3d_extraction.log
mkdir -p "$(dirname "$log_path")"

wait_for_idle_gpu() {
  local stable=0
  while (( stable < 3 )); do
    local sample free_mib utilization
    sample=$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n 1)
    free_mib=$(awk -F, '{gsub(/ /, "", $1); print $1}' <<<"$sample")
    utilization=$(awk -F, '{gsub(/ /, "", $2); print $2}' <<<"$sample")
    if (( free_mib >= 16384 && utilization <= 25 )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    printf '%s waiting_for_gpu free_mib=%s utilization=%s stable=%s/3\n' \
      "$(date --iso-8601=seconds)" "$free_mib" "$utilization" "$stable" >>"$log_path"
    if (( stable < 3 )); then
      sleep 60
    fi
  done
}

run_with_resume() {
  local stream=$1
  local checkpoint=$2
  local checkpoint_sha=$3
  local split=$4
  local output_root=$5
  local attempt=1
  while (( attempt <= 3 )); do
    wait_for_idle_gpu
    printf '%s start stream=%s split=%s attempt=%s\n' \
      "$(date --iso-8601=seconds)" "$stream" "$split" "$attempt" >>"$log_path"
    if PYTHONUNBUFFERED=1 python -m elsc.features.i3d \
      --video-root "$video_root" \
      --checkpoint "$checkpoint" \
      --expected-checkpoint-sha256 "$checkpoint_sha" \
      --stream-name "$stream" \
      --output-root "$output_root" \
      --temporal-metadata-root "$temporal_root" \
      --splits "$split" \
      --stride 1 \
      --batch-size 8 \
      --min-free-disk-gib 20 \
      --min-free-gpu-gib 16 \
      >>"$log_path" 2>&1; then
      printf '%s complete stream=%s split=%s\n' \
        "$(date --iso-8601=seconds)" "$stream" "$split" >>"$log_path"
      return 0
    fi
    printf '%s retry stream=%s split=%s attempt=%s\n' \
      "$(date --iso-8601=seconds)" "$stream" "$split" "$attempt" >>"$log_path"
    attempt=$((attempt + 1))
  done
  printf '%s failed stream=%s split=%s after=3_attempts\n' \
    "$(date --iso-8601=seconds)" "$stream" "$split" >>"$log_path"
  return 1
}

agnostic_checkpoint=artifacts/pretrained/bsl5k.pth.tar
agnostic_sha=6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f
aware_checkpoint=artifacts/pretrained/domain_aware_I3D_H2S.pth.tar
aware_sha=99e101d696ff63131b5d44fa6e465201216604ba5d8cc773f3cefa4a96ebd518

release_attempt=1
while (( release_attempt <= 3 )); do
  wait_for_idle_gpu
  printf '%s start official_release_test_parity attempt=%s\n' \
    "$(date --iso-8601=seconds)" "$release_attempt" >>"$log_path"
  if PYTHONUNBUFFERED=1 python -m elsc.release_eval \
    --config configs/ph_base.yaml \
    --output artifacts/release_eval/ph_test \
    --device cuda:0 >>"$log_path" 2>&1; then
    printf '%s complete official_release_test_parity\n' \
      "$(date --iso-8601=seconds)" >>"$log_path"
    break
  fi
  release_attempt=$((release_attempt + 1))
done
if (( release_attempt > 3 )); then
  printf '%s failed official_release_test_parity after=3_attempts\n' \
    "$(date --iso-8601=seconds)" >>"$log_path"
  exit 1
fi

wait_for_idle_gpu
printf '%s start i3d_release_feature_parity\n' "$(date --iso-8601=seconds)" >>"$log_path"
PYTHONUNBUFFERED=1 python -m elsc.features.validate_release \
  --device cuda:0 \
  --batch-size 8 \
  --min-free-disk-gib 20 \
  --min-free-gpu-gib 16 >>"$log_path" 2>&1
printf '%s complete i3d_release_feature_parity\n' "$(date --iso-8601=seconds)" >>"$log_path"

for split in dev train; do
  run_with_resume domain_agnostic "$agnostic_checkpoint" "$agnostic_sha" "$split" \
    artifacts/sign_features/ph_domain_agnostic
  run_with_resume domain_aware "$aware_checkpoint" "$aware_sha" "$split" \
    artifacts/sign_features/ph_domain_aware
done

printf '%s all_feature_extraction_complete\n' "$(date --iso-8601=seconds)" >>"$log_path"

python -m elsc.prepare --config configs/ph_base.yaml --splits train dev test \
  >>"$log_path" 2>&1
python -m elsc.audit --config configs/ph_base.yaml --stage assets \
  >>"$log_path" 2>&1

wait_for_idle_gpu
python -m elsc.audit --config configs/ph_base.yaml --stage parity \
  >>"$log_path" 2>&1

run_training() {
  local config=$1
  local run_dir=$2
  if [[ -f "$run_dir/selection.json" ]]; then
    printf '%s training_already_complete run_dir=%s\n' \
      "$(date --iso-8601=seconds)" "$run_dir" >>"$log_path"
    return 0
  fi
  wait_for_idle_gpu
  if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
    python -m elsc.train --config "$config" --run-dir "$run_dir" \
      --resume "$run_dir/checkpoints/last.pt" --device cuda:0 >>"$log_path" 2>&1
  else
    python -m elsc.train --config "$config" --run-dir "$run_dir" \
      --device cuda:0 >>"$log_path" 2>&1
  fi
}

run_training configs/ph_base.yaml runs/ph_base_s42
wait_for_idle_gpu
python -m elsc.evaluate --run-dir runs/ph_base_s42 --split dev --checkpoint best_dev \
  --device cuda:0 >>"$log_path" 2>&1

mkdir -p artifacts/campaign
python -m elsc.configure_stage \
  --template configs/ph_min.yaml \
  --teacher-run runs/ph_base_s42 \
  --output artifacts/campaign/ph_min_s42.yaml >>"$log_path" 2>&1

if [[ ! -f artifacts/cache/ph_train_elsc_v1/cache_meta.json ]]; then
  wait_for_idle_gpu
  python -m elsc.mining.build_cache \
    --config artifacts/campaign/ph_min_s42.yaml --split train --device cuda:0 \
    >>"$log_path" 2>&1
fi

run_training artifacts/campaign/ph_min_s42.yaml runs/ph_min_s42
wait_for_idle_gpu
python -m elsc.evaluate --run-dir runs/ph_min_s42 --split dev --checkpoint best_dev \
  --device cuda:0 >>"$log_path" 2>&1
python -m elsc.report \
  --baseline-runs runs/ph_base_s42 \
  --method-runs runs/ph_min_s42 \
  --split dev \
  --output artifacts/campaign/ph_min_vs_base_dev_s42.json >>"$log_path" 2>&1

printf '%s seed42_dev_screen_complete\n' "$(date --iso-8601=seconds)" >>"$log_path"
