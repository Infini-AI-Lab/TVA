# `logsigmoid_fwd_bf16`

**The kernel.** `logsigmoid_fwd_kernel` of fla v0.6.0 (`source/logsigmoid.py`, kernel lines
92-103 with the helpers `exp`, `log` of lines 53-56; host `logsigmoid_fwd` lines 106-121). One
program per `B` consecutive elements of the flattened `bfloat16` tensor `x` (`T = x.numel()`
elements in rows of `D = x.shape[-1]`, rows at stride `stride_x_row`): element `k` is column
`k % D` of row `k // D`; it computes `(min(0, x) - log(1 + exp(-|x|))) / temperature` in `float32`
and stores it rounded to `bfloat16` in element `k` of a fresh contiguous `y`. The claim is
`Claims.LogsigmoidFwd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
`k < T`, with `v` the initial `bfloat16` input at row `k / D`, column `k % D` (element
`xIdx d k = (k / D)·stride_x_row + k % D` of `x`) and `t` the host's double `temperature`:
- `y[k]` is a finite `bfloat16` within
  `yTol v t = (2^-8 + 2^-19)·|lsRef v / t| + 2^-23 / |t| + 2^-133` of `lsRef v / t`, where
  `lsRef v = -log(1 + e^-v)` is the exact log-sigmoid (`log σ(v)`);
- every byte outside the first `T` elements of `y` is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `B` a power of two, at most `2^20` (Triton's limits; every configuration of fla's autotune list);
- fewer than `2^31` programs, `cdiv(T, B) < 2^31` (CUDA's grid limit; it gives `T < 2^51`);
- `stride_x_row < 2^63` (an integer argument);
- the buffers hold every element read (rows `i < N` at `i·stride_x_row`, any stride: contiguous,
  `chunk` halves, broadcast rows) and written (`T` elements of `y`), and are valid in the initial
  memory; the written elements of `y` share no byte with the read elements of `x`;
- `temperature` a finite double with `2^-126 ≤ |t| ≤ 2^127` (its `float32` is a normal number);
- every input read is a finite `bfloat16` with `|v| + 1 ≤ 2^125·|t|` (the quotient stays below
  `2^126`, where `Device.DivFull` holds; for `|t| ≥ 4` every finite `bfloat16` qualifies).

**The launches covered** (`SETUP.md`): fla's test launches and the GLA-sized and long-sequence ones
(`setup_*_ok`). `mk_ok` covers the host's tensors, contiguous or a `chunk(2, dim=-1)` half, for
every `N ≥ 1` rows, `N·D ≤ 2^31`, and every temperature `TOk` admits; `Inst.ok` itself covers any
shape and stride within the conditions, `T = 0` (an empty grid) included.

**Device facts** the tolerance rests on (all in `Device P`): `Device.ExpF32` (at `X = |v|`),
`Device.LogF32`, `Device.DivFull`, and the totality facts for `exp`, `log` and `/`.

## Tolerances

Units: `u = 2^-24` (one `float32` rounding, `add_err`/`sub_err`: `u·|q| + 2^-150`), `2^-8` and
`2^-134` for the rounding to `bfloat16` (`cvt_err`). Write `a = |v|`, `E = e^-a`,
`ℓ = log(1 + E)`, `m = min(0, v)`; then `lsRef v = m - ℓ` (`lsRef_eq`), `ℓ ≤ |lsRef v| = ℓ - m`,
and `E ≤ 2ℓ` (`log(1 + E) ≥ E/(1 + E)`).

1. `tl.exp(-a)` (`ExpF32` at `X = a`, room since `E ≤ 1`): `e` within `(2^-22 + 2^-23·a)·E +
   2^-140`, and `a·E ≤ 1/e < 3/8`, so within `2^-22·E + 3·2^-26 + 2^-140`: relative to `ℓ`
   `2^-21`, plus an **absolute** `3·2^-26`.
2. `z = fl(1 + e)`: one rounding; `|log z - log(1 + e)| ≤ u·(1 + 2^-20)`, **absolute** (this is
   the kernel's own error for large `v`, where `1 + e^-v` rounds to `1`: `log1p` would be
   relative); `|log(1 + e) - ℓ| ≤ |e - E|·(1 + 2^-22)` (both arguments are above `1 - 2^-23`).
3. `L = tl.log(z)` (`LogF32`, `z ∈ [1/2, 3]`): `2^-22·|log z| + 2^-149`.
   Together (`L_err`): `|L - ℓ| ≤ 2^-20·ℓ + α₀`, `α₀ = 2^-24·113/64` (the exact sum is
   `3·2^-22·ℓ + 1.7500005·2^-24 + 2^-138`).
4. `s = fl(m - L)` (`m` exact: `tl.minimum` of `+0` and the exact widening of `v`; `-|x|` exact):
   one rounding, so (`s_err`) `|s - lsRef v| ≤ (2^-20 + 2^-23)·|lsRef v| + αs`,
   `αs = 2^-24·57/32`.
5. `t₃₂ = float32(t)`: `|t₃₂ - t| ≤ 2^-24·|t| + 2^-150 ≤ 2^-23·|t|` (`|t| ≥ 2^-126`).
   `q = s / t₃₂` (`DivFull`; `|s / t₃₂| ≤ 2^126` by `|v| + 1 ≤ 2^125·|t|`, `div_room`):
   `|s/t₃₂ - lsRef v/t| ≤ ((2^-20 + 2^-22)·R + αs/|t|)·(1 + 2^-22)` with `R = |lsRef v / t|`, and
   the division adds `2^-22·|s/t₃₂| + 2^-148`.
6. `c = bfloat16(q)` (`|q| ≤ 2^127`, `q_room`): `2^-8·|q| + 2^-134`.

Collecting (`final`): `|c - lsRef v / t| ≤ (2^-8 + 1.51·2^-20)·R + 1.789·2^-24/|t| + 2^-134 +
2^-148·(1 + 2^-8)`, below `yTol = (2^-8 + 2^-19)·R + 2^-23/|t| + 2^-133` (margins `1.32×` on the
`float32` part of the relative term, `1.12×` on the absolute term). The relative term is the
`bfloat16` rounding: the GPU reaches `0.995` of `yTol`. The absolute term matters only where
`|lsRef v|` is below about `2^-15`, that is `v > 10` (the median of `yTol/|ref|` on `randn·3`
inputs is `2^-8`).

## Exclusions

Each is a limit of the kernel, outside real use:
- **`|v| + 1 > 2^125·|t|`**: the quotient leaves `Device.DivFull`'s range, and near `2^128` it
  overflows: `v = -2^127` at `t = 0.25` gives `-∞` (exact `-6.8·10^38`, beyond `bfloat16`'s
  range anyway). For `|t| ≥ 4` no finite input is excluded.
- **`|t| < 2^-126`** (a subnormal or zero `float32` temperature): `t = 1e-40` gives `-∞`; a zero
  temperature divides by zero. **`|t| > 2^127`**: the `float32` may round to `±∞`.
- **Non-finite inputs**: `-∞` gives `-∞`, `+∞` gives `0`, NaN gives NaN (each the kernel's,
  outside the claim's finite inputs).
- **`cdiv(T, B) ≥ 2^31`**: CUDA's grid limit.
- **A `float32` `x`, an integer `temperature`, `FLA_USE_FAST_OPS=1`**: other compiled kernels (a
  different pointer dtype, an `int32` scalar divided after conversion, libdevice's fast `exp` and
  `log`), not this claim's model.

`check_gpu.py` (one H200) checks `Post` on the setup's launches and on edge launches.
