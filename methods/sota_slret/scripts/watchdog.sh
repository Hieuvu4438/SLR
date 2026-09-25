#!/usr/bin/env bash
# Keep a run alive: relaunch (resume) if its trainer process vanished without DONE.json. Max 10 restarts.
# Usage: watchdog.sh RUN_DIR [train.py args...]   (run detached: setsid nohup watchdog.sh ... &)
OUT=$1; shift
n=0
while [ ! -f "$OUT/DONE.json" ] && [ $n -lt 10 ]; do
  sleep 120
  if ! pgrep -f -- "--out $OUT " > /dev/null && [ ! -f "$OUT/DONE.json" ]; then
    if grep -q "Traceback" "$OUT.stdout" 2>/dev/null && [ "$(grep -c Traceback "$OUT.stdout")" -gt "${TB:-0}" ]; then
      TB=$(grep -c Traceback "$OUT.stdout"); echo "$(date) traceback seen; relaunching anyway ($n)" >> "$OUT.watchdog"
    fi
    n=$((n+1)); echo "$(date) relaunch $n" >> "$OUT.watchdog"
    /home/haipd/SLR/methods/sota_slret/scripts/launch.sh "$OUT" "$@" >> "$OUT.watchdog" 2>&1
  fi
done
echo "$(date) watchdog exit (done=$( [ -f "$OUT/DONE.json" ] && echo y || echo n ), restarts=$n)" >> "$OUT.watchdog"
