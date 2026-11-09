#!/bin/bash
# lr-32a: re-run the batch's evidence on GPU 0, one after another; each output kept here.
T=$1   # TVA's root
export CUDA_VISIBLE_DEVICES=0
for e in front-end-round32 num-dot-int8-round32 num-special-f32 desc-strides tut10-formats; do
  timeout 2400 "$T/.venv/bin/python" -I "$T/evidence/$e/run.py" > "rerun_$e.out" 2>/dev/null
  echo "exit $?" >> "rerun_$e.out"
done
timeout 3000 "$T/.venv/bin/python" -I "$T/evidence/num-dot-chunk16/run.py" --quick > rerun_num-dot-chunk16-quick.out 2>/dev/null
echo "exit $?" >> rerun_num-dot-chunk16-quick.out
