# `swiglu_fwd_bf16`

**The kernel.** `swiglu_fwd_kernel` of flash-linear-attention (fla) v0.6.0
(`source/swiglu.py`, kernel lines 92-102; host `swiglu_fwd` lines 105-124). The SwiGLU
activation, elementwise: `z = x · sigmoid(x) · y`, computed in `float32` from two `bfloat16`
inputs of one shape and rounded to `bfloat16`. The kernel walks the `T = x.numel()` elements as a
flat array, `B` per program; flat element `k` is column `k % D` of row `k // D` (`D =
x.shape[-1]`), read at `row · stride + col` in each input (so one half of a `chunk(2, dim=-1)` is
read in place) and written at `row · D + col` in the fresh contiguous `z`. The claim is
`Claims.SwigluFwd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
row `r < N` (`N = T / D`) and column `c < D`, with `a = x[r, c]` and `b = y[r, c]` read from the
initial memory, `z[r, c]` is a finite `bfloat16` and
- for `a > -89`: `|z[r, c] − zRef| ≤ zTol = (2^-8 + 2^-13)·|zRef| + 2^-133`, where
  `zRef = a · b / (1 + e^-a)` is the exact value;
- for `a ≤ -89`: `z[r, c] = 0` (`tl.exp(-a)` overflows to `+∞`, so the kernel's sigmoid is `+0`);

and every byte outside the first `N·D` elements of `z` is unchanged (so `x` and `y` are
unchanged).

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ D < 2^31` (`D` is the kernel's divisor; real row lengths are below `2^17`);
- `B` a power of two, at most `2^20` (Triton's block limit; fla's autotune list is `512` to `8192`,
  the host here passes `4096`);
- fewer than `2^31` programs, `cdiv(N·D, B) < 2^31` (CUDA's grid limit); it gives `T < 2^51`, so
  the `int64` index arithmetic does not wrap;
- the row strides of `x` and `y` below `2^63` (`int32` or `int64` arguments); any strides (rows
  apart, overlapping or broadcast with stride `0`);
- the buffers hold every element read (`r·stride + D ≤ len` for each row) or written (`N·D`), are
  valid, and the written elements of `z` share no byte with the elements of `x` and `y` read (the
  host allocates `z`; `x` and `y` may overlap each other);
- every input element read is a finite `bfloat16`, and every exact result has `|zRef| ≤ 2^127`
  (half of `bfloat16`'s largest value; beyond, the `float32` product or the rounding may overflow
  to `±∞`).

**The launches covered** (`SETUP.md`): the source's test shapes (contiguous and both halves of a
`chunk`), fla's `GatedMLP` at hidden sizes 2048 and 4096 on training batches and one token, every
`B` of fla's autotune list, and a batch of `2^32` elements (`T` an `int64` argument)
(`setup_*_ok`). `mk_ok` covers the host's tensors, contiguous or split, for every `N`, `1 ≤ D <
2^31` with `N·D ≤ 2^39`, and `B = 2^9 … 2^20`.

**Device facts** the tolerance rests on (all in `Device P`): `Device.ExpF32` (taken at
`X = |a|`), `Device.ExpF32Overflow`, `Device.DivFull`, `Device.DivF32InfDivisor`, and the totality
facts for `exp` and `/` on `float32`.

## Tolerances

Units: one `float32` rounding is `2^-24` relative and `2^-150` absolute (`FloatV.add_err`,
`mul_err`); the rounding to `bfloat16` `2^-8` and `2^-134` (`cvt_err`); the widenings
`bfloat16 → float32` and the negation are exact. Write `a`, `b` for the inputs, `e = e^-a`,
`σ = 1/(1 + e)`, so `zRef = a·σ·b`.

**`a > -89`.** The `bfloat16` values in `[64, 128)` are `1/2` apart, so `a ≥ -88.5`.
1. `E = tl.exp(-a)`, `Device.ExpF32` at `X = |a|`: room holds (`e ≤ e^88.5 ≤ 0.82·2^128`), and
   `|E − e| ≤ (2^-22 + 2^-23·|a|)·e + 2^-140 ≤ 3·2^-18·(1 + e) + 2^-140`: for `a < 0`,
   `|a| ≤ 88.5` gives `90.5·2^-23·e`; for `a ≥ 0`, `e ≤ 1` and `a·e ≤ 1`.
2. `D = fl(1 + E)`: one rounding, `|D − (1 + e)| ≤ 0.8·2^-16·(1 + e)`.
3. `S = 1 / D`, `Device.DivFull` (`2^-22` relative, `2^-148` absolute; `1/D < 2`): since
   `1 + e ≤ 1 + e^88.5 ≤ 2^128`, `σ ≥ 2^-128` and the absolute term is at most `2^-20·σ`; with step
   2, `|S − σ| ≤ ((2^-22 + c)/(1 − c) + 2^-20)·σ ≤ 2^-16·σ`, `c = 0.8·2^-16`.
4. `P = fl(a·S)`: for `a ≠ 0`, `|a| ≥ 2^-133` (the smallest `bfloat16` subnormal) and `|a|·σ ≥
   2^-135` (`σ ≥ 1/4` when `|a| ≤ 1`; `|a|·σ ≥ σ ≥ 2^-128` otherwise), so the absolute `2^-150` is
   at most `2^-15·|a|σ`, and `|P − aσ| ≤ (2^-16 + 2^-24(1 + 2^-16) + 2^-15)·|aσ| ≤ 2^-14·|aσ|`.
   For `a = 0`, `P`, the next product and the output are zeros, and `zRef = 0`.
5. `Z = fl(P·b)`: `|Z − zRef| ≤ (2^-14 + 2^-24(1 + 2^-14))·|zRef| + 2^-150`.
6. `O = bfloat16(Z)`: `|O − zRef| ≤ (2^-8 + (2^-14 + 2^-24(1 + 2^-14))(1 + 2^-8))·|zRef| + 2^-134 +
   2^-150(1 + 2^-8) ≤ (2^-8 + 2^-13)·|zRef| + 2^-133`.

Room in each format: `|a·S| ≤ 255·2^120·(1 + 2^-16) ≤` `float32`'s `maxVal`; `|P·b| ≤
(1 + 2^-14)·2^127 ≤ maxVal`; `|Z| ≤ (1 + 2^-13)·2^127 + 2^-150 ≤ 255·2^120`, `bfloat16`'s `maxVal`
(by `Pre`'s `|zRef| ≤ 2^127`). So every result is finite.

**`a ≤ -89`.** `-a ≥ 89` is past `88.72283935546875`, so `tl.exp(-a) = +∞`
(`Device.ExpF32Overflow`); `1 + ∞ = +∞` (IEEE); `1 / +∞ = +0` (`Device.DivF32InfDivisor`); the
exact products `a·0` and `0·b` are `0`, so each rounds to a zero (within `2^-150` of `0`, below the
smallest subnormal), and so does the `bfloat16` rounding: `z = ±0`. The exact value is not zero:
`|zRef| ≤ 89·e^-89·|b| < 2^-121.9·|b|`; that is the kernel's own limit (the sigmoid
`1/(1 + e^-a)` underflows), and the claim states what the kernel gives.

The relative tolerance is the `bfloat16` rounding (`2^-8`) and `2^-13` for the `float32` steps;
the GPU reaches `0.966` of it (a rounding to nearest at the bottom of a binade). Rounding toward
zero, or a `bfloat16` rounding of `x·sigmoid(x)` before the second product, exceeds it.

## Exclusions

Each is a limit of the kernel or outside the real launches:
- **`|zRef| > 2^127`**: the `float32` product `P·b` or the rounding to `bfloat16` may overflow to
  `±∞` (the rounding of an exact value beyond `bfloat16`'s range, or near it); real activations
  are far below.
- **Non-finite inputs**: outside the real launches; the kernel then propagates NaN or `±∞`.
- **`D = 0`** (an empty last dimension): the kernel's constexpr divisor is `0`; no program runs.
- **`cdiv(T, B) ≥ 2^31`**: CUDA's grid limit.
- **Other dtypes**: `float16` or `float32` inputs, or `x` and `y` of different dtypes (each its own
  compiled kernel), are not part of this claim.
- The backward kernel (`swiglu_fwdbwd_kernel`) and fla's Ascend backend (`triton_ascend.py`) are
  other kernels.

The kernel uses no feature outside the library's subset: no reduction, no atomics, no loop.

`check_gpu.py` (one H200) checks `Post` on the setup's launches and on edge launches.
