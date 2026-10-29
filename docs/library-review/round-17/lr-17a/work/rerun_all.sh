#!/bin/bash
# lr-17a: re-run the round-17 evidence and the reviewers' work the batch cites (GPU 0).
set -u
ulimit -c 0
T=$(cd ../../../../.. && pwd)
PY=$T/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
for name in hint-unread-round17 dead-code-round17 cdiv-round17; do
  $PY -I $T/evidence/$name/run.py > rerun-$name.out 2>&1
done
$PY -I $T/docs/library-review/round-15/lr-15a/work/probes.py > rerun-r15a-probes.out 2>&1
$PY -I $T/docs/library-review/round-15/lr-15b/work/fe_edges.py > rerun-r15b-fe_edges.out 2>&1
$PY -I $T/docs/library-review/round-15/lr-15b/work/dead_after_return.py > rerun-r15b-dead_after_return.out 2>&1
$PY -I $T/docs/library-review/round-16/lr-16b/work/graph_capture.py > rerun-r16b-graph_capture.out 2>&1
echo done > rerun.done
