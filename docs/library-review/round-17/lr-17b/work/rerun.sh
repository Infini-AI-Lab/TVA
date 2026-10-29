#!/bin/bash
# Re-runs the evidence and the earlier reviewers' probes the round-17 batch cites.
# Run from this work folder: bash rerun.sh <tva-root>   (TMPDIR set by the caller)
set -u
ROOT="$1"
PY="$ROOT/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
ulimit -c 0
run() {  # name, script
  echo "== $1 $(date -u +%H:%M:%S)" >&2
  "$PY" -I "$2" > "rerun-$1.out" 2> "rerun-$1.err"
  echo "   exit $?" >&2
}
run hint-unread-round17 "$ROOT/evidence/hint-unread-round17/run.py"
run dead-code-round17   "$ROOT/evidence/dead-code-round17/run.py"
run lr15a-probes        "$ROOT/docs/library-review/round-15/lr-15a/work/probes.py"
run lr15b-fe_edges      "$ROOT/docs/library-review/round-15/lr-15b/work/fe_edges.py"
run lr15b-dead_after_return "$ROOT/docs/library-review/round-15/lr-15b/work/dead_after_return.py"
CACHE=$(mktemp -d)
TRITON_CACHE_DIR="$CACHE" run lr16b-graph_capture "$ROOT/docs/library-review/round-16/lr-16b/work/graph_capture.py"
rm -rf "$CACHE"
run cdiv-round17        "$ROOT/evidence/cdiv-round17/run.py"
echo "== done $(date -u +%H:%M:%S)" >&2
