# `tl.dot_scaled` on sm_90 (Triton's emulation)

Library evidence (not a kernel's), for library issue L4. Triton 3.8.0 (a local virtualenv, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question.** sm_90 has no microscaling instructions, so Triton emulates `tl.dot_scaled`. How
accurate is it for MXFP8 (`"e4m3"` by `"e4m3"`) and for MXFP8 by MXFP4 (`"e4m3"` by `"e2m1"`),
with `e8m0` scales in groups of 32 along `K` and both operands packed along `K`? The library
states `DotScaledE4m3` and `DotScaledE4m3E2m1` ([rules/num.md](../../docs/rules/num.md),
"`tl.dot_scaled` on sm_90"), for either `fast_math`: per output element, when every scaled value
of its row of `lhs` and column of `rhs` is `0` or has a magnitude in `[2^-126, bfloat16 max]`
(`scaledDomBF16`) and every scale byte is in `1 … 254` (`e8m0Dom`), the result is finite and
within `(K+2)·2^-22·mag + (K+2)·2^-126` of `acc + Σ va·vb` (`mag = |acc| + Σ|va·vb|`), with room in
`float32`.

**What Triton compiles it to** (`primitives/triton-3.8.0-DecomposeScaledBlocked.cpp:191-228`,
and the PTX in `out.txt`): each element is converted to `bfloat16` (exact for `e4m3`, through
`float16`, and for `e2m1`), its scale is decoded to `bfloat16` by a bit shift and multiplied in
(`mul.bf16`, `mul.bf16x2`), and the products go to a `bfloat16` dot into `float32`
(`wgmma.mma_async….m64n64k16.f32.bf16.bf16` for a `64×64` block).

**Method.** Both formats, both `fast_math` values, four block shapes (`16×16` with 4 warps,
`64×64` with 4, `128×128` with 8, `128×64` with 4), `K = 32 … 512`. Six kinds of inputs, two
draws each: random element bits (no NaN encodings) with scales around `2^0`, scales over most of
the range, scales near the bottom of `bfloat16`'s normal range, products near the top of
`float32`'s range, a large accumulator with all bits set, and same-sign elements with every
fraction bit set (`1.875`, `e2m1` `6.0`) under one scale per row and column. The exact values are
decoded from the bits (`e4m3` through torch's `float8_e4m3fn`, `e2m1` by its table, `e8m0` as
`2^(s−127)`) and summed in `float64`; `(K+1)·2^-52·mag` is added to the error for the
reference's rounding. Elements whose row or column has a scaled value outside `scaledDomBF16`,
or no room, are not checked; the script counts them. Then one element with each excluded scale
byte, and the PTX.

**Result** (`out.txt`).

| `rhs` format | Elements checked / computed (all shapes, `K`, kinds, `fast_math`) | Outside the bound | Largest error / bound |
|---|---|---|---|
| `e4m3` | 3131020 / 3471360 | 0 | 0.108 |
| `e2m1` | 3161591 / 3471360 | 0 | 0.100 |

The largest ratio grows slowly with `K` (`0.081` at `K = 32`, `0.108` at `K = 512` for `e4m3`)
and is the same for both `fast_math` values and every block shape.

Excluded scale bytes (`a = b = 1.0`, scale of `b` `1.0`): `0x00` gives `0` (the OCP value is
`2^-127`); `0xFF` gives NaN; `0x01`, `0x7F`, `0xFE` give `2^-126`, `1`, `2^127`, as decoded.

**Conclusions.**
1. Both facts hold on every element checked, for both `fast_math` values and every shape and
   `K` tested, with margin: no error above `0.11` of the bound.
2. The emulation is a `bfloat16` `tl.dot` after an exact scaling, so the facts take `DotBF16`'s
   constants. On `scaledDomBF16` every scaled value is `0` or a normal `bfloat16` value, so that dot
   has no subnormal operand and its lifted magnitudes are `|a·b|`. The 25-bit window of
   [evidence/num-dot16-model](../num-dot16-model/README.md) (its bit-level model, equal to
   the device on every element tested, and its derivation, which covers every `K`) then bounds the
   dot by `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K`, and its Part G3 checks these facts on window
   inputs (at most `0.0057` of the bound). (Until round 26 this conclusion cited `evidence/num-dot`'s
   per-chunk derivation, "every term down to `2^-23` of the largest", which round 25 found holds only
   for normal operands, as here; library issue L129.)
3. The domain is needed: outside `scaledDomBF16` the `bfloat16` multiplication by the scale can
   underflow or overflow. Scale `0x00` is read as `0`, not `2^-127`, and `0xFF` is NaN, so
   `e8m0Dom` excludes both.
