#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/SLR

download_dir=artifacts/downloads
archive_path=$download_dir/sign_features.zip
log_path=artifacts/logs/sign_features_download.log
expected_bytes=2797376661
reserve_bytes=$((20 * 1024 * 1024 * 1024))
available_bytes=$(df -B1 --output=avail "$download_dir" 2>/dev/null | tail -n 1 | tr -d ' ' || true)
if [[ -z "$available_bytes" ]]; then
  mkdir -p "$download_dir"
  available_bytes=$(df -B1 --output=avail "$download_dir" | tail -n 1 | tr -d ' ')
fi
mkdir -p "$download_dir" "$(dirname "$log_path")"

if (( available_bytes < expected_bytes + reserve_bytes )); then
  printf 'insufficient disk: available=%s required=%s\n' \
    "$available_bytes" "$((expected_bytes + reserve_bytes))" >>"$log_path"
  exit 1
fi

url='https://drive.usercontent.google.com/download?id=1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy&export=download&confirm=t'
printf '%s starting expected_bytes=%s available_bytes=%s\n' \
  "$(date --iso-8601=seconds)" "$expected_bytes" "$available_bytes" >>"$log_path"

/home/haipd/.local/bin/aria2c \
  --continue=true \
  --max-connection-per-server=16 \
  --split=16 \
  --min-split-size=1M \
  --file-allocation=none \
  --connect-timeout=30 \
  --timeout=60 \
  --retry-wait=5 \
  --max-tries=0 \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --console-log-level=notice \
  --summary-interval=30 \
  --dir="$download_dir" \
  --out=sign_features.zip \
  "$url" >>"$log_path" 2>&1

actual_bytes=$(stat -c %s "$archive_path")
if (( actual_bytes != expected_bytes )); then
  printf 'size mismatch: actual=%s expected=%s\n' "$actual_bytes" "$expected_bytes" >>"$log_path"
  exit 1
fi
unzip -tq "$archive_path" >>"$log_path" 2>&1
sha256sum "$archive_path" >"$archive_path.sha256"
printf '%s verified bytes=%s sha256=%s\n' \
  "$(date --iso-8601=seconds)" "$actual_bytes" "$(cut -d' ' -f1 "$archive_path.sha256")" \
  >>"$log_path"
