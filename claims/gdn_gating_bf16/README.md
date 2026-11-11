# `gdn_gating_bf16`

**The kernel.** `fused_gdn_gating_kernel` of SGLang v0.5.23 (`source/fused_gdn_gating.py`, kernel
lines 53-84; host `fused_gdn_gating` lines 87-119). One program per (token, block of 8 heads): it
loads `A_log`, `a`, `b`, `dt_bias`, stores `g = -exp(A_log) · softplus(a + dt_bias)` (softplus with
`beta = 1`, `threshold = 20`) and `beta_output = sigmoid(b)` (rounded to `bfloat16`, stored as
`float32`). The claim is `Claims.GdnGating.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
token `i < batch` and head `h < num_heads`, with `A = A_log[h]`, `x = a[i, h] + dt_bias[h]` (the
exact real sum of the two `bfloat16` inputs) and `bb = b[i, h]`, read from the initial memory:
- `g[i·num_heads + h]` is a finite `float32` within
  `gTol A x = 2^-23·(64 + min(|A|, 128))·(e^A + 2^-121)·(spRef x + 2^-6)` of
  `gRef A x = -e^A · spRef x`, `spRef x = ln(1 + e^x)` for `x ≤ 20` and `x` above; or it is `-∞`
  and `maxVal ≤ |gRef| + gTol` (the exact value is at the top of `float32`'s range);
- `beta_output[i·num_heads + h]` is a finite `float32` within `2^-7·σ(bb) + 2^-133` of
  `σ(bb) = 1 / (1 + e^-bb)`;
- every byte outside the two outputs' first `batch·num_heads` elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `batch < 2^31` and `cdiv(num_heads, 8) < 2^16` (CUDA's grid limits); `batch ≤ 2^22` (a context
  length);
- the `int32` offsets do not wrap: `batch·num_heads ≤ 2^31`,
  `(batch − 1)·stride + num_heads ≤ 2^31` for `a` and `b`, strides below `2^31`;
- the buffers hold every element read or written (any row stride for `a` and `b`, so contiguous
  rows and the halves of one `[batch, 2·num_heads]` projection), and are valid in the initial
  memory;
- `g` shares no byte with the inputs or `beta_output`, and `beta_output` none with the inputs;
- every input read is finite, `A_log[h] ≤ 88.72282` and `b[i, h] ≥ -88.72282` (`ExpRoom`: `tl.exp`
  does not overflow), and the `float32` sum `a + dt_bias` does not round to `+∞`.

**The launches covered** (`SETUP.md`): the host's 9 test cases and SETUP.md's Qwen3-Next and
Qwen3.5 configurations (`setup_*_ok`). `mk_ok` covers the host's tensors, contiguous or split, for
every `1 ≤ batch ≤ 2^22`, `num_heads ≤ 2^16`, `batch·num_heads ≤ 2^26`.

**Device facts** the tolerances rest on (all in `Device P`): `Device.ExpF32`,
`Device.ExpF32NegInf` (a sum `a + dt_bias` that rounds to `-∞`), `Device.LogF32`,
`Device.DivFull`, and the totality facts for `exp`, `log` and `/`.

## Tolerances

Units: `U = 2^-23`, `u = 2^-24` (one `float32` rounding, `add_err`/`mul_err`: `u·|q| + 2^-150`).
The device facts used: `Device.ExpF32` at `X` (`rel 2^-22 + 2^-23·X`, `abs 2^-140`, with room),
`ExpF32NegInf`, `LogF32` (`2^-22`, `2^-149`), `Device.DivFull` (`2^-22`, `2^-148`), the IEEE `+ * -` and
conversions (`FloatV.add_err`, `mul_err`, `cvt_err`; the products by `1.0` and the negation are
exact). Write `xf = fl(x)` (the `float32` sum, `|xf - x| ≤ u|x| + 2^-150`), `S` the kernel's softplus,
`E = tl.exp(A)`.

**Softplus, `xf ≤ 20`** (`S = log(fl(1 + exp(xf)))`).
1. `tl.exp(xf)` with `X = |xf|`: within `(2 + |xf|)U·e^xf + 2^-140`; room holds since `xf ≤ 20`
   (and for `xf ≪ 0`, `(1 + 2^-23|xf|)e^xf` is tiny). Its effect on `ln(1 + ·)` is at most
   `ε·e^x/(1+e^x)/(1-ε)`, so relative to `sp(xf)` at most `(2 + |xf|)U·σ/sp`: `≤ 1.69U` for
   `xf ∈ [0, 20]` (`max (2+x)σ(x)/sp(x) = 1.683`), `≤ (2 + |xf|)U` for `xf ∈ [-20, 0)`, and an
   absolute `≤ 22U·e^-20 < 2^-47` below `-20`. `xf = -∞` gives `+0` (`ExpF32NegInf`), then
   `log(1) ∈ [-2^-149, 2^-149]`.
2. `1 + exp`: one rounding, `|δ| ≤ u`, so `|ln fl(1+E') - ln(1+E')| ≤ u(1 + 2^-20)`: **absolute**,
   whatever `x` (this is the kernel's own error for `x ≪ 0`, where `1 + e^x` rounds to `1`).
3. `tl.log`: `2U` relative and `2^-149`.
4. `x → xf`: `|sp(xf) - sp(x)| ≤ σ(ξ)·|xf - x|`; for `x ≥ 0`, `σ ≤ 1` and `|x| ≤ sp(x)`: `0.5U`
   relative; for `x ∈ [-20, 0)`, `σ ≤ sp`: `0.5|x|U` relative; below `-20`: absolute `< 2^-48`.
   At the threshold (`xf ≤ 20 < x` or the reverse, only within `20u` of `20`):
   `|ln(1+e^20) - 20| < e^-20`, under `2^-32` relative.
Sum: relative `ρ_S ≤ (5.2 + 1.5·m)U` with `m = min(max(-x, 0), 20)`, absolute
`α_S ≤ 2^-24·(1 + 2^-19)`.

**Softplus, `xf > 20`** (`S = xf`): `0.5U` relative (plus the threshold case above).

**`g = fl(-E·S)`**. `E` with `X = |A|` (room by `Pre`): `|E - e^A| ≤ r_A e^A + 2^-140`,
`r_A = (2 + |A|)U`. Then
`|E·S - e^A·sp| ≤ e^A·sp·(ρ_S + r_A(1 + ρ_S)) + e^A·α_S·(1 + r_A) + 2^-140·(sp·(1 + ρ_S) + α_S)`,
and the product's rounding adds `u·|E·S| + 2^-150`. Collecting:
- relative: `ρ_S + r_A + 0.5U + (cross terms) ≤ (7.7 + 1.5m + |A|)U` for `|A| ≤ 2^20`, against
  `gFine`'s `(16 + 2m + |A|)U` (below) (a factor `≥ 2.08` on the constant part, `1.33` on the `m` part);
  for `|A| > 2^20`, `e^A·sp ≤ e^(-2^20)·2^128`, far below `2^-148`;
- absolute: `e^A·α_S·(1 + r_A)(1 + u) ≤ 1.13·2^-24·e^A < 2^-23·e^A` (`r_A ≤ 1/8` for
  `A ≥ -2^20`; `A ≤ 88.73` gives `r_A < 2^-16`); `2^-140·sp·(1 + ρ_S)(1 + u) ≤ 2^-139·sp`; the rest
  (`2^-140·α_S`, `2^-150`, and the `A < -2^20` terms) `≤ 2^-148`.
- overflow: when `fl(-E·S)` is `-∞`, `|E·S| ≥ maxVal`, and `|E·S| ≤ |gRef| + gFine` by the same
  bounds (`S ≥ -2^-148`, `E > 0` up to `2^-140`, so the sign is negative and `+∞` is impossible).

So the fine bound

`gFine A x = (2^-19 + 2^-23·(|A| + 2m))·|gRef| + 2^-23·e^A + 2^-139·spRef x + 2^-148`,
`m = min(max(-x, 0), 20)`,

is provable from the cited facts.

**Relaxation `gFine ≤ gTol`.** Write `a = min(|A|, 128)`, `sp = spRef x > 0`, `U = 2^-23`, and
expand `gTol = U(64 + a)·(e^A·sp + 2^-6·e^A + 2^-121·sp + 2^-127)`. Every term of `gFine` is below
one term of `gTol`:
- `|A| ≤ 128` (so `a = |A|`; this includes every `A ≤ 88.73` that `Pre` allows on the positive
  side): the relative term is `U(16 + |A| + 2m)·e^A·sp ≤ U(56 + |A|)·e^A·sp ≤ U(64 + a)·e^A·sp`
  (`m ≤ 20`); `U·e^A = U·64·2^-6·e^A ≤ U(64 + a)·2^-6·e^A`; `2^-139·sp ≤ U·64·2^-121·sp = 2^-138·sp`;
  `2^-148 ≤ U·64·2^-127 = 2^-144`.
- `A < -128` (`a = 128`): `(56 + |A|)·e^A` decreases in `|A|`, so the relative term is at most
  `U·184·e^-128·sp < 2^-200·sp`; the term `U·192·2^-121·sp = 2^-136.4·sp` of `gTol` covers it and
  `2^-139·sp` together; the other two terms as above.
The overflow clause then follows from `|gRef| + gFine ≤ |gRef| + gTol`.

**`beta_output`**: `tl.exp(-b)` (`X = |b|`, room by `Pre`): relative `≤ (2 + 88.73)U < 1.1·10^-5`
on `σ` (as above, `≤ ε` relative), `1 +`: `u`, `DivFull`: `2^-22` and `2^-148`, then the `bfloat16`
rounding: `2^-8` relative and `2^-134` (its grid below `2^-126` is `2^-133`), and the widening back
is exact. Total relative `2^-8 + 1.2·10^-5 < 2^-7`, absolute `2^-134 + 2^-148·(1 + 2^-8) < 2^-133`.
The GPU reaches `0.498` of it.

## Exclusions

Each is a limit of the kernel, outside real use:
- **`A_log > 88.72282`**: `exp(A_log)` overflows to `+∞`, and `g` is `-∞` or NaN where the exact
  value is finite (real `A_log = log(A)` with `A ∈ (0, 16]`).
- **`a + dt_bias ≥ 2^128 − 2^103`**: the `float32` sum rounds to `+∞`, and `g = -∞` where the
  exact `g` may be finite.
- **`b < -88.72282`**: `exp(-b)` overflows and `beta_output = 0`, outside the tolerance while
  `σ(b)` is a `bfloat16` subnormal above `2^-133`.
- **`batch·num_heads > 2^31`** or **`(batch − 1)·stride + num_heads > 2^31`**: an `int32` offset
  wraps.
- **`batch ≥ 2^31` or `cdiv(num_heads, 8) ≥ 2^16`**: CUDA's grid limit.
- **Strides `≥ 2^31`** (Triton would pass an `int64`), **non-finite inputs**, **`batch > 2^22`**:
  outside the file's launches and real shapes.

`check_gpu.py` (one H200) checks `Post` on the setup's launches and on edge launches.
