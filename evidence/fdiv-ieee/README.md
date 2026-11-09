# `tl.fdiv(x, y, ieee_rounding=True)` and `tl.softmax(…, ieee_rounding=True)` (library issue L166)

Library evidence (not a kernel's), for library issue L166 (from library review lr-35b, round 35).
Triton 3.8.0 (TVA's `.venv`, torch 2.11.0+cu130, numpy 2.3.5), NVIDIA H200 NVL (sm_90), GPU 0,
2026-11-08. Script: [run.py](run.py); output: [out.txt](out.txt) (under a minute). Part E (library
issue L168, from library review lr-36b's `work/softmax_fmt.py`, round 36): [softmax_fmt.py](softmax_fmt.py),
[softmax_fmt.out.txt](softmax_fmt.out.txt) (GPU 0, 2026-11-08, under a minute). Part F (library
issue L168, round 38, from library review lr-37a's `work/fmt_probe.py`, round 37):
[fmt_probe.py](fmt_probe.py), [fmt_probe.out.txt](fmt_probe.out.txt) (GPU 0, 2026-11-08, under a
minute; identical to lr-37a's output).

**Question.** The library gave `/` and `tl.fdiv` with `ieee_rounding=False` a meaning (`tl.div`,
`Rules.Num.DivFull`, a product for a following addition, library issue L163), and `tl.div_rn` the
IEEE one, but said nothing of `tl.fdiv(x, y, ieee_rounding=True)`, and called `tl.softmax`, which
ends in `fdiv(num, den, ieee_rounding)`, a product whatever its `ieee_rounding`. What does
`ieee_rounding=True` compile to, and is it fused into a following addition?

**The source.** `semantic.py:331-341`: `fdiv` checks that both operands are floating, does not
convert them (`binary_op_type_checking_impl(…, arithmetic_check=False, …)`), and emits
`tt.precise_divf` when `ieee_rounding` is set, `arith.divf` otherwise. `math.py:203-210`:
`tl.div_rn` (`float32` only) emits the same `tt.precise_divf`. `nvidia-ElementwiseOpToLLVM.cpp:886-887`
lowers it to `llvm.nvvm.div.rn.f`, a `float32` intrinsic. `standard.py:54-64`: `tl.softmax` is
`fdiv(exp(x - max(x)), sum(…), ieee_rounding)`.

**Method** (`run.py`).
- A. The PTX of `tl.fdiv(x, y, ieee_rounding=True)`, `tl.div_rn(x, y)` and `tl.fdiv(x, y)` on
  `float32`, `float16`, `bfloat16` and `float64` operands, and of `tl.softmax` with each
  `ieee_rounding`, on `float32`.
- B. `float32` quotients of `tl.fdiv(…, ieee_rounding=True)`, `tl.div_rn` and the default
  `tl.fdiv`, bit for bit (NaNs as NaN), against each other and against the IEEE quotient (numpy
  `float32` division): every pair of 256 bit patterns (zeros, `±1`, `3`, `0.1`, `-7`, `±inf`, NaN,
  `2^126`, `2^127`, the smallest normal and subnormal, the largest finite, `1 + 2^-23`, `2^±100`,
  the rest random), and `2^24` random pairs.
- C. `tl.fdiv(x, y, ieee_rounding=True) + c` and `c - tl.fdiv(x, y, ieee_rounding=True)`, with
  `enable_fp_fusion` on (the default), on `2^20` inputs where `c` nearly cancels the quotient, so
  that a multiply fused with the addition would give another result on almost every input: each
  result against the IEEE sum of the IEEE quotient; the SASS's counts of `FFMA`, `FADD`, `FMUL`,
  `MUFU.RCP` and calls.
- D. `tl.softmax(x, dim=1, keep_dims=True, ieee_rounding=True)` on 64 tiles of `64 × 16`
  (standard normal data scaled by 4 or 30), against `num / den` with the IEEE division, `num` and
  `den` stored by the same kernel.
- E (`softmax_fmt.py`, library issue L168). `tl.softmax(x, dim=1, keep_dims=True, ieee_rounding=r)`
  on a `64 × 16` tile of `float16`, `bfloat16`, `float32` and `float64`, each `r`: whether it
  compiles, the result's dtype (a `tl.static_assert`), the PTX division, and the results bit for bit
  against `num / den` with numpy's IEEE division in the result's format, `num` and `den` stored by
  the same kernel; `tl.fdiv(x, Y, ieee_rounding=True)` with a `float32` `x` and a Python constant `Y`
  (`3.0`, `1.0`), and with a `float16` or `float64` divisor; `tl.softmax(x, ieee_rounding=True) - c`
  with fusion on, on 16 tiles where `c` nearly cancels the result, against the IEEE difference.
- F (`fmt_probe.py`, library issue L168, round 38). `tl.softmax(x, dim=1, keep_dims=True,
  ieee_rounding=r)` on a `64 × 16` tile of `float8_e5m2` (`float8e5`), `float8_e4m3fn`
  (`float8e4nv`), `int32` and `int16`, each `r`: whether it compiles, the result's width (stored by
  the kernel) and the PTX division; `tl.sigmoid` on `float16`, `bfloat16`, `float32`, `float64` and
  the two fp8 formats: whether it compiles, the PTX division, and whether `tl.sigmoid(x) + c` differs
  with fusion on and off.

**Result** (`out.txt`).
- A. `tl.fdiv(ieee_rounding=True)` and `tl.div_rn` on `float32`: `div.rn.f32`, one
  `tt.precise_divf`; the default `tl.fdiv`: `div.full.f32`. On `float16` and `bfloat16` operands all
  three fail to compile (`tl.div_rn` in the front end; `tl.fdiv` with either `ieee_rounding` in the
  lowering, since nothing converts the operands); on `float64`, `tl.fdiv(ieee_rounding=True)` fails
  to compile ("intrinsic call returns double but `llvm.nvvm.div.rn.f` actually returns float"), as
  `tl.div_rn` does, and the default `tl.fdiv` is `div.rn.f64`. `tl.softmax(ieee_rounding=False)`
  holds `div.full.f32`, `tl.softmax(ieee_rounding=True)` `div.rn.f32`.
- B. `ieee_rounding=True` and `tl.div_rn` differ on 0 of 65536 and 0 of 16777216 pairs, and from the
  IEEE quotient on 0 and 0; the default `tl.fdiv` differs from the IEEE quotient on 13677 and
  3718265 (within `DivFull`, [evidence/num-div-f32](../num-div-f32/README.md)).
- C. 0 of 1048576 results differ from the IEEE sum of the IEEE quotient, in both forms, where an
  unrounded quotient would give another result on 1048575. (The SASS's `FFMA`s are those of
  `div.rn.f32`'s own Newton steps, with its slow path a call; the addition is an `FADD`.)
- D. 0 of 65536 results differ from `num / den` with the IEEE division.
- E. On `float16`, `bfloat16` and `float32` tiles the result of `tl.softmax` is `float32` (`tl.max`
  widens floats narrower than 32 bits, so `x - tl.max(x)` is `float32`, and so are its `tl.exp`, the
  `tl.sum` and the quotient): with `ieee_rounding=True` it holds `div.rn.f32` and differs from the
  IEEE `num / den` on 0 of 1024 elements; with the default, `div.full.f32` (257, 220 and 213 of 1024
  differ, within `DivFull`). On `float64` the result is `float64`: the default holds `div.rn.f64` (0 of
  1024 off the IEEE quotient), and `ieee_rounding=True` fails to compile ("failed to translate module
  to LLVM IR"). `tl.fdiv(x, 3.0 or 1.0, ieee_rounding=True)`: `div.rn.f32`, 0 of 256 off IEEE; a
  `float16` or `float64` divisor of a `float32` `x` fails to compile (`tt.precise_divf` requires one
  type). `tl.softmax(ieee_rounding=True) - c`: 0 of 16384 off the IEEE difference.
- F. On `float8e5` and `float8e4nv` tiles `tl.softmax` compiles and its result is 32 bits wide
  (`float32`, as for the 16-bit formats): `div.full.f32` by default, `div.rn.f32` with
  `ieee_rounding=True`. On `int32` and `int16` tiles it fails to compile with either `ieee_rounding`
  (`tl.exp` takes only `float32` and `float64`, `math.py:94-98`). `tl.sigmoid` compiles on `float32`
  (`div.full.f32`) and `float64` (`div.rn.f64`) only; on `float16`, `bfloat16` and the fp8 formats it
  fails to compile.

**Conclusions.** On `float32` operands `tl.fdiv(x, y, ieee_rounding=True)` is `tl.div_rn`
(`FloatV.divRn`, IEEE), and on any other format it has no outcome (a compile error). It is not a
product for the fusion rule: its following addition is the IEEE `tl.add`. On a `float8e5`,
`float8e4nv`, `float16`, `bfloat16` or `float32` tensor `tl.softmax` computes and returns `float32`, and ends in `tl.div_rn`
with `ieee_rounding=True` and in `/` (`tl.div`, a product) with the default; on a `float64` tensor
it ends in the IEEE `float64` `/` with the default (not a product), and with `ieee_rounding=True` it
has no outcome (a compile error); on an integer tensor it has no outcome (a compile error). Writing the
IEEE forms' additions as fusable ones (`tl.addc`) is sound but loose, since the fusable relation
contains the IEEE result. Library review lr-35b's `work/fusion_sweep.py` (round 35) found the same
for `tl.fdiv(ieee_rounding=True)` and `tl.softmax(ieee_rounding=True)` with fusion on and off.
