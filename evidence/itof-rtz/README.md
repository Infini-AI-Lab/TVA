# An int-to-float value narrowed toward zero

Library evidence (not a kernel's), for library issue L24. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Since round 6** (library issue L36), the fact is stated in [SEMANTICS.md](../../docs/SEMANTICS.md) (`x.to(float)`);
TRANSCRIPTION.md is guidance.

**Question.** The transcription rule `[itof-narrow]` made every narrowing of an int-to-float value
one `tl.itof` to the final format. Does that hold when the narrowing has
`fp_downcast_rounding="rtz"`, and what does the device compute then?

**What the library reads.**
- MLIR folds `truncf(sitofp x)` into `sitofp x` only when the `truncf` has the default rounding
  mode (`primitives/llvm-ArithCanonicalization.td:453-463`).
- `cast` emits a narrowing float conversion with a rounding mode other than to-nearest-even as
  `tt.fp_to_fp` with that mode, not as `arith.truncf` (`primitives/triton-3.8.0-semantic.py:822-845`):
  no fold applies to it. `fp_downcast_rounding="rtne"` given explicitly is the default (`truncf`).

**Method.** Every `int32` `x` (all `2^32`, in chunks of `2^26`) through
`x.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz")` and the same into `tl.float16`,
with the default-rounding chains as controls. References, computed on the GPU with torch:
- round-4 rule: the `float32` nearest to `x` (torch's conversion, to nearest even), then toward
  zero: the high half of the pattern for `bfloat16`; for `float16` the nearest value stepped one
  place toward zero when it is larger in magnitude (so an overflow gives 65504);
- round-3 rule: `tl.ItofRel`, one rounding of `x` to nearest even (computed exactly in integer
  arithmetic) or two through `float32` (an `int32` is exact in `float64`, so through `float64` is
  the single rounding).

**Result** (`out.txt`).

| Chain | vs the round-4 rule | vs the round-3 rule (`tl.ItofRel`) |
|---|---|---|
| `.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz")` | 0 of `2^32` differ | 2147431680 outside it |
| `.to(tl.float32).to(tl.float16, fp_downcast_rounding="rtz")` | 0 differ | 4294894593 outside it |
| the same chains with the default rounding (controls) | | 0 outside it, for both formats |

The TTIR of the kernel has two `tt.fp_to_fp` (the rtz narrowings) and no `arith.truncf` (the
default narrowings were folded into `sitofp`). Example: `x = -2147483583` gives `-2139095040` with
rtz, while the nearest `bfloat16` is `-2147483648`.

**Conclusions.**
1. With `fp_downcast_rounding="rtz"`, an int-to-float value narrowed to `bfloat16` or `float16` is
   the conversion to the intermediate format (one rounding to nearest even), then a conversion
   toward zero: the model writes `tl.itof` to the intermediate format, then `tl.cvt … .rz`. The
   round-3 rule was wrong for about half of the `int32` values in `bfloat16` and nearly all in
   `float16` (most overflow, where toward zero gives the largest finite value).
2. With the default rounding the round-3 rule (one `tl.itof` to the final format) holds on every
   `int32`.
