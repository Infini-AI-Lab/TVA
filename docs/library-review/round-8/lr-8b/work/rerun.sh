#!/bin/bash
# Re-run the round-8 evidence scripts; outputs go to this folder.
ulimit -c 0
: "${TMPDIR:?set TMPDIR to a scratch folder outside TVA}"
export TMPDIR
export CUDA_VISIBLE_DEVICES=0
T=../../../../..
PY=$T/.venv/bin/python
for e in hint-loops hint-rank2 runtime-if scalar-args desc-scratch autotune-launch compiled-launch other-writers; do
  echo "== $e $(date -u +%H:%M:%S)"
  $PY -I $T/evidence/$e/run.py > rerun-$e.txt 2>&1
  echo "exit $?"
done
echo "== promotion $(date -u +%H:%M:%S)"
$PY -I $T/evidence/promotion/table.py > rerun-promotion-table.tsv 2> rerun-promotion-table.err
echo "exit $?"
$PY -I $T/evidence/promotion/device.py > rerun-promotion-device.txt 2>&1
echo "exit $?"
$PY -I $T/evidence/promotion/constants.py > rerun-promotion-constants.txt 2>&1
echo "exit $?"
echo "== done $(date -u +%H:%M:%S)"
