#!/bin/bash
# On lovelace: run TVA's device evidence scripts on the L40s. Jobs on one GPU run one after
# another; different GPUs run in parallel. Logs and a done list in ~/tva-l40/tva/logs.
cd ~/tva-l40/tva; ln -sfn ~/tva-l40/py .venv; mkdir -p logs; export TMPDIR=~/tva-l40/tmp; mkdir -p $TMPDIR
J=( "0 evidence/num-math-f32/run.py" "1 evidence/num-special-f32/run.py" "2 evidence/num-cvt8/run.py"
    "2 evidence/num-cvt8/specials.py" "3 evidence/num-atomic-add/run.py" "3 evidence/num-atomic-add/sem_scope.py"
    "4 evidence/num-div-f32/run.py" "5 evidence/num-dot/run.py" "5 evidence/num-dot/ieee_ratio.py"
    "6 evidence/num-dot16-model/run.py" "7 evidence/num-dot-int8/run.py" "7 evidence/reduce-tree/run.py"
    "8 evidence/num-dot-chunk16/run.py" "8 evidence/num-dot-chunk16-order/run.py" "9 evidence/num-dot-chunk/run.py"
    "9 evidence/num-dot-chunk-loss/run.py" "1 evidence/num-dot-scaled/run.py" )
for g in 0 1 2 3 4 5 6 7 8 9; do
  ( for j in "${J[@]}"; do set -- $j; [ "$1" = "$g" ] || continue
      n=$(echo $2 | tr / _); s=$(date +%s)
      CUDA_VISIBLE_DEVICES=$g .venv/bin/python -I $2 > logs/$n.log 2>&1; rc=$?
      echo "$2 gpu$g rc=$rc $(( $(date +%s)-s ))s" >> logs/done.txt
    done ) &
done
wait; echo ALL >> logs/done.txt
