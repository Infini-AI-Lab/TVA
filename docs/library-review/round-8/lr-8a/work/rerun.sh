#!/bin/bash
# Re-runs the round-8 evidence scripts. Usage: rerun.sh <output folder>
# Run from this work folder; TVA's root is five levels up.
set -u
OUT="$1"
WORK="$(cd "$(dirname "$0")" && pwd)"
TVA="$(cd "$WORK/../../../../.." && pwd)"
PY="$TVA/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
ulimit -c 0
cd "$WORK"
for d in hint-loops hint-rank2 runtime-if scalar-args desc-scratch autotune-launch compiled-launch other-writers; do
  start=$(date +%s)
  "$PY" -I "$TVA/evidence/$d/run.py" > "$OUT/$d.txt" 2> "$OUT/$d.err"
  echo "$d exit=$? secs=$(( $(date +%s) - start ))" >> "$OUT/status.txt"
done
start=$(date +%s)
"$PY" -I "$TVA/evidence/promotion/table.py" > "$OUT/promotion-table.tsv" 2> "$OUT/promotion-table.err"
echo "promotion-table exit=$? secs=$(( $(date +%s) - start ))" >> "$OUT/status.txt"
"$PY" -I "$TVA/evidence/promotion/rules.py" > "$OUT/promotion-rules.txt" 2>&1
echo "promotion-rules exit=$?" >> "$OUT/status.txt"
"$PY" -I "$TVA/evidence/promotion/device.py" > "$OUT/promotion-device.txt" 2> "$OUT/promotion-device.err"
echo "promotion-device exit=$?" >> "$OUT/status.txt"
"$PY" -I "$TVA/evidence/promotion/constants.py" > "$OUT/promotion-constants.txt" 2> "$OUT/promotion-constants.err"
echo "promotion-constants exit=$?" >> "$OUT/status.txt"
echo done >> "$OUT/status.txt"
