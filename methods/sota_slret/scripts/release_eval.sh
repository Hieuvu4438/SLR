#!/usr/bin/env bash
# Baseline RELEASE PARITY: evaluate official SEDS checkpoints with the untouched upstream entrypoint.
# Usage: release_eval.sh {ph|csl|h2s} OUTDIR
set -euo pipefail
DS=$1; OUT=$2
source /home/haipd/miniconda3/bin/activate seds
cd /home/haipd/SLR/third_party/SEDS
COMMON="--signbert --fusion_type gloss_atten --rgb_pose_match --rgb_pose_match_loss 0.4 --lr 1e-5 --sign_lr 1e-4 \
 --max_words 32 --feature_len 64 --max_length_frames 300 --slide_windows 16 --windows_stride 1 \
 --crop_size 256 --frames_threshold 0.1 --threshold 0.4 --batch_size_val 64 --coef_lr 1. --freeze_layer_num 0 \
 --linear_patch 2d --sim_header Filip --pretrained_clip_name ViT-B/32 --num_thread_reader 8"
case $DS in
 ph)  EXTRA="--data_path data_ph --features_path ./PHOENIX-2014-T/RTM_Keypoints/ --features_RGB_path ./PHOENIX-2014-T/I3D_features/ --datatype ph_pose --init_model ckpts/ph_best_model.bin";;
 csl) EXTRA="--data_path data_csl --features_path ./CSL/RTM_Keypoints/ --features_RGB_path ./CSL/I3D_features/ --datatype csl_pose --original_size 512 --init_model ckpts/csl_best_model.bin";;
 h2s) EXTRA="--data_path data_h2 --features_path ./How2Sign/RTMpose/Pose_all_24rates/ --features_RGB_path ./How2Sign/I3D_features/ --datatype h2s_pose --original_size_w 256 --original_size_h 256 --init_model ckpts/h2s_best_model.bin";;
esac
PORT=$((29500 + RANDOM % 400))
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port $PORT main_task_retrieval.py --do_eval --output_dir "$OUT" $COMMON $EXTRA
