#!/bin/bash
# Re-run the device evidence the batch cites (Triton 3.8.0, GPU 0). Run from the TVA root.
# Outputs: docs/library-review/round-6/lr-6b/work/rerun/<topic>.out.txt, and a diff against out.txt.
OUT=docs/library-review/round-6/lr-6b/work/rerun
export CUDA_VISIBLE_DEVICES=0
for t in config-cond debug-calls pure-operands literals; do
  start=$(date +%s)
  .venv/bin/python -I evidence/$t/run.py > $OUT/$t.out.txt 2> $OUT/$t.err.txt
  rc=$?
  end=$(date +%s)
  if diff -q evidence/$t/out.txt $OUT/$t.out.txt > /dev/null; then same=identical; else same=differs; fi
  echo "$t rc=$rc $((end-start))s output $same" >> $OUT/status.txt
done
.venv/bin/python -I evidence/literals/int_full.py > $OUT/literals-int_full.out.txt 2> $OUT/literals-int_full.err.txt
rc=$?
if diff -q evidence/literals/int_full.out.txt $OUT/literals-int_full.out.txt > /dev/null; then same=identical; else same=differs; fi
echo "literals/int_full rc=$rc output $same" >> $OUT/status.txt
