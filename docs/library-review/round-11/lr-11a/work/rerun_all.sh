#!/bin/bash
# lr-11a: re-run the round-11 evidence scripts and compare with the recorded output.
# set TMPDIR to a private temporary folder before running
ulimit -c 0
V=../../../../../.venv/bin/python
E=../../../../../evidence
for d in hint-multi python-numbers if-names py-builtins compiled-options launch-frame claim-satisfiable; do
  CUDA_VISIBLE_DEVICES=0 $V -I $E/$d/run.py > rerun-$d.txt 2>&1
  if diff -q rerun-$d.txt $E/$d/output.txt >/dev/null; then echo "$d IDENTICAL"; else echo "$d DIFFERS"; fi
done
