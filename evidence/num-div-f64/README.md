# `/` on `float64` (`DivRn P .binary64`)

Library evidence (not a kernel's), for library issue L67. Triton 3.8.0 (TVA's Python environment,
torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0; Lean 4.35.0; 2026-10-28. Script:
[run.py](run.py); output: [out.txt](out.txt).

**Question.** The library states `DivRn P .binary64` ([rules/num.md](../../docs/rules/num.md),
"Division"): every result of `x / y` on `float64` is `FloatV.divRn x y`, IEEE division with one
rounding to nearest even (`Format.div .rn`), up to NaN payload. Triton lowers `arith.divf` on
`f64` to `llvm.nvvm.div.rn.d` (`primitives/triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:569-571`),
which is `div.rn.f64` in the PTX ([out.txt](out.txt), last section). Does the device give
exactly the library's quotient?

**Method.** The library side is the definition itself, not a re-implementation: the native
executable of [evidence/float-layer](../float-layer/README.md) (`build.sh` compiles
`lean/TVA/Math/Defs.lean` with its `Driver.lean`; spec `div:f64`, `Format.div .rn` on `binary64`).
`run.py` writes the input pairs as records, runs the same pairs through a Triton kernel on the
device, and both sides reduce each chunk of `2^20` records to one 64-bit number
`h = Σ out(i)·(2iK + 1) mod 2^64` (`K = 0x9E3779B97F4A7C15`, odd, so one differing output always
changes `h`), every NaN output first replaced by the canonical NaN. A chunk whose numbers differ
is dumped on both sides and its differing inputs listed (none did). The kernel loads `x` and `y`
as `float64` bit patterns and stores `x / y`; a second kernel stores `tl.fdiv(x, y)`, and a third
`x / c` for a Python constant `c` (a `tl.constexpr`, built as the `float64` `c` itself). The
families (fixed seeds, so a re-run gives the same pairs):

| Family | Pairs | What |
|---|---|---|
| `uniform` | `2^26` | both operands uniform 64-bit patterns (NaNs, infinities, zeros and subnormals at their natural rates) |
| `struct` | `2^26` | as evidence/float-layer's records: a quarter uniform patterns, a quarter short significands (exact quotients, ties), a quarter structured (random exponents, the divisor's near the dividend's), a quarter special values |
| `near1` | `2^25` | divisors within `2^20` ulps of `1` or `−1` (both sides), dividends finite with any exponent |
| `binade1` | `2^25` | divisors in `[1/2, 2)` with random (or short) significands |
| `subnormal-dividend` | `2^25` | subnormal dividends, divisors near `1` (exponent within `±60`) or any finite |
| `subnormal-quotient` | `2^26` | normal operands whose quotient is subnormal or just above `2^-1022` (the divisor's exponent is the dividend's plus `1021` to `1075`) |
| `subnormal-divisor` | `2^24` | subnormal divisors (quotients up to overflow) |
| `pow2` | `2^24` | divisors that are powers of two: exact quotients, and shifts into the subnormal range with ties |
| `overflow` | `2^24` | quotients near `maxVal` and beyond (the exponents differ by `1021` to `1026`) |
| `near-exact` | `2^25` | `x = RN(q·y)` for random `q`, `y`: the quotient lies within about an ulp of `q`, so the rounding decides between `q` and its neighbours |
| `midpoint` | `2^22` | `x = RN(mid·y)` for `mid` halfway between two consecutive `float64` values (exact arithmetic): the quotient is within a fraction of an ulp of a midpoint, on either side |
| `specials` | `36²` | every pair of 36 special values: `±0`, the smallest and largest subnormals, the smallest normal, `1`, `1 ± ulp`, `maxVal`, `±∞`, NaNs |
| `struct, tl.fdiv` | `2^24` | `tl.fdiv(x, y)` (`ieee_rounding=False`) on `struct` pairs |
| constants | `11 × 2^22` | `x / c` on `struct` dividends, `c` ∈ {`1.0`, `-1.0`, `2.0`, `0.5`, `3.0`, `0.1`, `7.0`, `2^-1074`, `2^1023`, `1e308`, `1/3`} |

**Result** ([out.txt](out.txt)). 452,986,128 pairs: every family agrees, on every chunk, with the
library's `Format.div .rn`. Every kernel divides with `div.rn.f64`, except `x / 1.0`, which the
compiler folds to `x` (the IEEE quotient). With `tl.fdiv` and with a constant divisor the result is
the same IEEE quotient.

**Conclusions.**
1. `/` on `float64` (and `tl.fdiv`) is IEEE division on the H200 NVL, bit for bit up to NaN
   payload, on every pair tested: subnormal dividends, quotients and divisors, divisors near `1`,
   powers of two, overflow, quotients next to the exact value and next to midpoints, and the
   special values. So `DivRn P .binary64` holds, with the library's own `FloatV.divRn` as the
   reference.
2. Constant divisors are divided the same way (`div.rn.f64`), and the one fold, `x / 1.0 = x`, is
   exact.

**Re-run.** Build the driver once (from `evidence/float-layer`, `sh build.sh B` with a build folder
`B` outside the tree), then from this folder `CUDA_VISIBLE_DEVICES=0 python -I run.py B`
(`--jobs N` Lean processes, 32 by default; `--quick` a smoke test of `2^20` pairs per family into
`B/quick.txt`). It writes `out.txt` and takes about 12 minutes on 32 cores; the time per family
goes to standard error.
