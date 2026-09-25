#!/usr/bin/env bash
# Sequential experiment queue (run detached). Each line of QUEUE_FILE: RUN_DIR|train.py args
# Waits for WAIT_DONE (a DONE.json path) first. Each run is retried/resumed up to 5 times.
QF=$1; WAIT_DONE=$2
source /home/haipd/miniconda3/bin/activate seds; cd /home/haipd/SLR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
while [ -n "$WAIT_DONE" ] && [ ! -f "$WAIT_DONE" ]; do sleep 60; done
while IFS='|' read -r OUT ARGS; do
  [ -z "$OUT" ] && continue; [[ "$OUT" == \#* ]] && continue
  for try in 1 2 3 4 5; do
    [ -f "$OUT/DONE.json" ] && break
    echo "$(date) start $OUT try $try" >> "$QF.log"
    python methods/sota_slret/src/train.py --out "$OUT" $ARGS >> "$OUT.stdout" 2>&1 < /dev/null
    sleep 10
  done
  echo "$(date) end $OUT done=$( [ -f "$OUT/DONE.json" ] && echo y || echo n )" >> "$QF.log"
done < "$QF"
echo "$(date) queue finished" >> "$QF.log"
