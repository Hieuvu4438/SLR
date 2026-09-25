#!/usr/bin/env bash
# Upstream SEDS training (untouched entrypoint) on our held-out-val split, single GPU.
# Usage: train_seds_baseline.sh DS SPLITDIR OUTDIR SEED EPOCHS [extra args...]
set -euo pipefail
DS=$1; SPLIT=$2; OUT=$3; SEED=$4; EP=$5; shift 5
source /home/haipd/miniconda3/bin/activate seds
cd /home/haipd/SLR/third_party/SEDS
COMMON="--signbert --init_sign_model ckpt/pretrain_signbert.pth --fusion_type gloss_atten --rgb_pose_match --rgb_pose_match_loss 0.4 \
 --lr 1e-5 --sign_lr 1e-4 --max_words 32 --feature_len 64 --max_length_frames 300 --slide_windows 16 --windows_stride 1 \
 --crop_size 256 --frames_threshold 0.1 --threshold 0.4 --batch_size_val 64 --coef_lr 1. --freeze_layer_num 0 \
 --linear_patch 2d --sim_header Filip --pretrained_clip_name ViT-B/32 --batch_size 128 --n_display 10 --num_thread_reader 20"
case $DS in
 ph)  EXTRA="--features_path ./PHOENIX-2014-T/RTM_Keypoints/ --datatype ph_pose";;
 csl) EXTRA="--features_path ./CSL/RTM_Keypoints/ --datatype csl_pose --original_size 512";;
 h2s) EXTRA="--features_path ./How2Sign/RTMpose/Pose_all_24rates/ --datatype h2s_pose --original_size_w 256 --original_size_h 256";;
esac
PORT=$((29500 + RANDOM % 400))
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port $PORT main_task_retrieval.py --do_train \
  --epochs $EP --seed $SEED --data_path $SPLIT/data --features_RGB_path $SPLIT/rgb --output_dir "$OUT" $COMMON $EXTRA "$@"
