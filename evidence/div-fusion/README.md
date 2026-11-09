# A float32 `/` is a product for the fusion rule (library issue L163)

Library evidence (not a kernel's), for library issue L163. Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-08. Script: [run.py](run.py); output:
[out.txt](out.txt) (a few minutes).

**Question.** SEMANTICS.md ("Float `+` and `-`") counted only `*`, `tl.dot` and `tl.rand` as products
whose following addition may fuse, and gave every other addition its IEEE meaning. On `float32`, `/`
is `div.full.f32` (docs/rules/num.md, "Division"), which `ptxas` expands into `MUFU.RCP` and a multiply
by the (scaled) dividend; with `enable_fp_fusion` that multiply may fuse with a following addition into
one `FFMA` (library review lr-34b, round 34). So the addition takes an unrounded product. Is every
result then one that `AddRel true u v` (`lean/TVA/Lang/Ops.lean`) allows, with `u` the quotient the
division alone gives (the value the model's `tl.div` picks, within `DivFull`) and `v` the other operand?
That is: the IEEE sum, or the rounding of `p + v` for an exact `p` that rounds to `u` (or of `u + p`
with `p` rounding to `v`), or an infinite operand.

**Method** (`run.py`). For each spelling below, a first launch stores the quotient alone (`q`, with the
same inputs and the same compile options), and a second launch stores the spelling's result `r`. Each
`r` is compared exactly (Python `Fraction`s, rounding to nearest even in the result's format, overflow
and both zeros included) with the set `AddRel true u v` gives for the operands the model writes: `u = q`
and `v = c` for `q + c`; `v = -c` for `q - c` (`tl.subc`); `u = c`, `v = -q` for `c - q`; the `where`'s
value for the `tl.where` form; both quotients for `x / y + z / w` and the `tl.sum`; the converted
quotient for a `.to` that converts. Counted: results outside the set, and results that are not the IEEE
sum of the operands (where the fusion shows). Regimes of 4096 elements each:
- *cancel*: `x`, `y` in `[0.5, 2)`, and `c` minus the quotient moved by up to 3 units in the last
  place (in `c`'s format), so that the unrounded product shows in the result;
- *random*: random finite `float32` patterns, `c` the quotient's magnitude times `2^k`, `|k| ≤ 30`;
- *divisor > 2^126* and *subnormal divisor*: the two scaled paths of `div.full.f32`;
- *subnormal quotient*;
- *special*: every combination of 14 values (`±0`, `±∞`, NaN, `±1`, `3`, the least subnormal,
  `2^127`, `1.5·2^126`, the largest finite value, `2^-126`, `0.75`) for `x`, `y` and `c`.

Spellings: `x / y + c`, `c + x / y`, `x / y - c`, `c - x / y`, `tl.where(m, x / y, 0.0) + c`,
`(x / y).to(tl.float32) + c`, `tl.fdiv(x, y) + c`, `tl.sigmoid(x) + c`, `x / 3.0 + c`, `1.0 / y + c`,
`x / y + z / w`, `tl.sum(tl.join(x / y, z / w), axis=1)`, `r = c; r += x / y` in a loop,
`-(x / y) + c`, `(x / y).to(tl.float64) + c` with a `float64` `c`, `(x / y).to(tl.float16) + c` with a
`float16` `c`, and `x / y + c` with `x`, `y`, `c` all `float16` or all `bfloat16` (the quotient is
`float32`, `c` is widened). A control runs `x / y + c` with `enable_fp_fusion=False`. The PTX and SASS
instructions are listed.

**Result** (`out.txt`).
- No result outside `AddRel`: 0 of 430080 (and 0 of 24576 in the control).
- 162200 results differ from the IEEE sum of the operands: every nearly cancelling element of the
  plain spellings (`+`, `-`, the `.to(tl.float32)` that changes nothing, `tl.fdiv`, a constant divisor,
  two quotients, the `tl.sum`, the loop), and most of the 16-bit ones. These are the fused results that
  the IEEE form of SEMANTICS.md before L163 did not allow (as lr-34b found, no `P.div` value with an
  IEEE addition gives them).
- The instructions: `div.full.f32` and `add.f32`/`sub.f32` in PTX become `MUFU.RCP` and `FFMA` in
  SASS. `tl.where(m, x / y, 0.0) + c` keeps an `FMUL` and an `FADD`, and the converting `.to` forms a
  `DADD` or `HADD2`: they are not fused, and every result is the IEEE sum. With
  `enable_fp_fusion=False`, `add.rn.f32` and `FADD`: every result is the IEEE sum.
- `tl.sigmoid(x) + c` and `1.0 / y + c` contain an `FFMA`, but their numerator is `1` (scaled by a
  power of two), so the product the addition could take unrounded is the reciprocal itself, and no
  result differs from the IEEE sum on the cancelling inputs.

**Conclusions.** A `float32` quotient (of `/` or `tl.fdiv` with `ieee_rounding=False`, also of 16-bit
or integer operands, and inside `tl.sigmoid` of a `float32` tensor and the default `tl.softmax` of a
`float8e5`, `float8e4nv`, `float16`, `bfloat16` or `float32` one, which divides in `float32`: library
issues L166, L168; the fp8 formats from [evidence/fdiv-ieee](../fdiv-ieee/README.md) part F) feeding
an addition is a product for the fusion rule: the
addition is `tl.addc`/`tl.subc` (`AddRel E.fusion`) on the quotient `tl.div` picks, and a reduction or
scan of quotients `tl.sumc`/`tl.cumsumc` (SEMANTICS.md, "Float `+` and `-`"). The quotient keeps its
own `DivFull` error, and the addition may then lose up to one rounding's bound of the quotient, as for
any product. `tl.div_rn` and `float64` `/` are IEEE and not fused (lr-34b's `x / y - q` stays `0` for
both), so they are not products.
