#!/bin/bash
# lr-3a: re-run the GPU evidence the round-3 batch cites, output into this work folder.
# Run from the TVA root: bash docs/library-review/round-3/lr-3a/work/rerun_gpu.sh
W=docs/library-review/round-3/lr-3a/work
export CUDA_VISIBLE_DEVICES=0
for t in intra-program pdl env-knobs int-reductions permute argmax pure-operands; do
  echo "== $t $(date -Is)"
  .venv/bin/python -I evidence/$t/run.py > $W/rerun_$t.out 2> $W/rerun_$t.err
  echo "exit $?"
done
