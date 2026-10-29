#!/bin/bash
# Re-run the GPU evidence the round-4 batch cites, on GPU 0, from the TVA root.
cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=0
W=docs/library-review/round-4/lr-4b/work
for t in config-cond literals launch-options pdl bcast-mul itof-rtz; do
  echo "== $t $(date -Is)"
  .venv/bin/python -I evidence/$t/run.py > $W/rerun_$t.out 2> $W/rerun_$t.err
  echo "   exit $?  $(date -Is)"
done
