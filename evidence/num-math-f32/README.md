# The approximate `float32` math functions on sm_90

Library evidence (not a kernel's), for library issue L4. Triton 3.8.0 (a local virtualenv, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question.** How accurate are `tl.exp`, `tl.math.exp2`, `tl.log`, `tl.math.log2`, `tl.sqrt` and
`tl.math.rsqrt` on `float32`, and what do they give at `-inf`? The facts the library states for
them (`TVA.Rules.Num.Sm90`, [rules/num.md](../../docs/rules/num.md)) must hold on every input.

**What Triton compiles them to** (the PTX of one kernel per function, in `out.txt`):
- `tl.exp`: a `mul.f32` by `log2 e` (rounded to `float32`), then `ex2.approx.f32`, without `.ftz`
  (`primitives/triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:634-658`, `ExpOpConversionApprox`);
- `tl.math.exp2`: libdevice's `__nv_exp2f`, which with Triton's default `enable_reflect_ftz`
  is `ex2.approx.ftz.f32` (results below `2^-126` are flushed);
- `tl.log`, `tl.math.log2`: libdevice's `__nv_logf`, `__nv_log2f`, a polynomial with
  `fma.rn.ftz.f32` that scales a subnormal input by `2^23` first;
- `tl.sqrt`: `sqrt.approx.ftz.f32`; `tl.math.rsqrt`: `rsqrt.approx.ftz.f32` (a subnormal input
  is flushed: `sqrt` gives `0`, `rsqrt` gives `+inf`).

**Method.** Every `float32` bit pattern (all `2^32`, in 64 chunks of `2^26`) through the six
functions on the device. The reference is the real function in `float64`, computed by numpy on the
host (glibc's `exp`, `exp2`, `log`, `log2` are within one `float64` ulp; `sqrt` is correctly
rounded). For each fact the script takes every finite input in the fact's domain whose reference
value leaves room in `float32` (`(1 + rel)·|ref| + abs ≤ maxVal`, `MathAcc`'s condition), checks
that the result is finite, and checks the bound with the reference's own error added:
`err·(1 + 2^-52) + 2^-51·|ref| ≤ rel·|ref| + abs`. For `ExpF32` (the bound grows with `X`) the
tightest instance for `x` is `X = |x|`, which the script uses.

| Fact | Domain | Bound |
|---|---|---|
| `ExpF32` | `x ∈ [-X, X]`, every `X ≥ 0` | `(2^-22 + 2^-23·X)·e^x + 2^-140` |
| `Exp2F32` | every `x` | `2^-22·2^x + 2^-126` |
| `LogF32` | `x > 0` | `2^-22·\|ln x\| + 2^-149` |
| `Log2F32` | `x > 0` | `2^-22·\|log₂ x\| + 2^-149` |
| `SqrtF32` | `x = 0` or `x ≥ 2^-126` | `2^-22·√x + 2^-149` |
| `RsqrtF32` | `x ≥ 2^-126` | `2^-22/√x + 2^-149` |

**Result** (`out.txt`).

| Function | Inputs where the fact applies | Outside the bound | Non-finite | Largest error / bound | Largest relative error (normal reference) |
|---|---|---|---|---|---|
| `tl.exp` | 3258020375 | 0 | 0 | 0.603 | `2^-17.99` (at `x = 88.14`) |
| `tl.math.exp2` | 3263168512 | 0 | 0 | 1.000, below 1 (flushed results; conclusion 3) | `2^-22.71` |
| `tl.log` | 2139095039 | 0 | 0 | 0.337 | `2^-23.57` |
| `tl.math.log2` | 2139095039 | 0 | 0 | 0.454 | `2^-23.14` |
| `tl.sqrt` | 2130706434 | 0 | 0 | 0.420 | `2^-23.25` |
| `tl.math.rsqrt` | 2130706432 | 0 | 0 | 0.523 | `2^-22.94` |

Special values: `exp(-inf)` and `exp2(-inf)` are `+0` (`0x00000000`) — the facts `ExpF32NegInf`
and `Exp2F32NegInf`; `exp(+inf) = exp2(+inf) = +inf`; `log`/`log2` of `±0` is `-inf`, of a
negative number NaN.

**Conclusions.**
1. Each of the six facts holds on every `float32` input it covers.
2. `tl.exp`'s relative error grows with `|x|` (to `2^-18` near the overflow at `x ≈ 88.7`): the
   product `x·log2 e` is rounded before `ex2.approx`. Its fact therefore has the relative
   constant `2^-22 + 2^-23·X` on `[-X, X]`; the largest ratio, `0.60`, is at small `|x|`, where
   `ex2.approx`'s own error dominates.
3. `tl.math.exp2` flushes results below `2^-126` to `0`. A flushed result's error is the
   reference itself, below `2^-126`, which is why the absolute constant is `2^-126`. The ratio
   approaches `1` there from below (`x` just below `-126`, reference just below `2^-126`, result
   `0`) and is not reached: for a reference below `2^-126` it is `2^x/(2^-22·2^x + 2^-126)`,
   below `1/(1 + 2^-22)`, `1.000` to three digits, and no input is outside the bound. It is the
   flush, bounded by construction, not a measurement near its limit. Every normal result is
   within `2^-22.71` relative. (An earlier version of this conclusion said the ratio `1.000` is "reached exactly";
   library issue L66.)
4. `tl.sqrt` and `tl.math.rsqrt` flush subnormal inputs, so their facts exclude them;
   `tl.sqrt(±0) = ±0`.
