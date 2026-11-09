# Claim: `l2norm_fwd_bf16`

The kernel is `l2norm_fwd_kernel` in `source/l2norm.py` (lines 44–60), launched
by its host `l2norm_fwd` (lines 63–90) with `BT = 16`, `num_warps = 4` on the grid `(cdiv(T, 16),)`.
The claim is `Claims.L2NormFwd.Statement` in `Claim.lean`: `∀ K P, Assumptions K P → Spec K`. The
file proves `consistent`, `satisfiable` and the coverage facts `setup_*_ok`; it does not prove
`Statement`.

## Specification

For every covered instance and initial memory, every launch ends normally, and for each row `t < T`,
with `a_j` the row's `bfloat16` inputs in the initial memory and
`rstdRef = 1 / sqrt(Σ_{j<D} a_j² + eps)` (exact, in the reals, with the host's double `eps`):

- `rstd[t]` is a finite `float32` with `|rstd[t] − rstdRef| ≤ 2^-12 · rstdRef`;
- each `y[t, j]`, `j < D`, is a finite value of `y`'s dtype with
  `|y[t, j] − a_j·rstdRef| ≤ 2^-7 · |a_j|·rstdRef + 2^-132` for a `bfloat16` `y`
  (`output_dtype = None`), and `≤ 2^-11 · |a_j|·rstdRef + 2^-149` for a `float32` `y`
  (`output_dtype = torch.float32`; `YOk`);
- every byte outside the first `T·D` elements of `y` and the first `T` elements of `rstd` is unchanged.

The reference is computed from the host's inputs (the initial memory of `x`, read at `t·D + j` of the
`[T, D]` view, and the host's `eps`), not from what the kernel loads or how it indexes; every output
is constrained, and the frame covers the rest.

## Conditions

`Inst.ok`:
- `1 ≤ D ≤ 512` (this kernel's path; the host asserts `D ≤ 512`); `BD` a power of two with
  `D ≤ BD ≤ 512` (the host's `BD = next_power_of_2(D)`);
- `BT` any power of two with `BT·BD ≤ 2^20` (Triton's largest block): the host's `16`, and every `BT`
  of the removed autotune list (8 … 128);
- `cdiv(T, BT) < 2^31` (CUDA's grid limit on axis 0); `T = 0` included (a zero grid ends normally,
  memory unchanged). `T` is a runtime `int32` below `2^31` and an `int64` from `2^31` (`do_not_specialize`
  keeps even `T = 1` a runtime scalar); the model takes both classes;
- `y` `bfloat16` or `float32` (`Inst.y : Sum (Buf .bf16) (Buf .f32)`; the model's `y` pointer is of
  that dtype);
- `x`, `y` of at least `T·D` elements, `rstd` of at least `T`; the written elements of `y` and `rstd`
  share no byte with each other or with the read elements of `x` (the host allocates both fresh);
- `eps` a finite double in `[2^-125, 2^126]` (the default `1e-6`; the kernel gets `FloatV.hostF32 eps`).

`Pre`: the three buffers are valid; every element the launch reads is a finite `bfloat16`; and on every
row `Σ_j x[t, j]² + eps ≤ 2^126` (the row's norm below `2^63`).

`Compiler.Faithful`, part (A), point by point: (4) the launch passes `num_warps` only, no
`@triton.heuristics`, no `debug`; (5) no descriptors; (6) a direct launch of the `@triton.jit` function,
not a `CompiledKernel`; (8) no host descriptor. The source has no
`@triton.autotune` (SETUP.md: removed). Part (B), the environment, is assumed: the claim relies on it.

## Assumptions

- `[rule] Compiler.Faithful` (`faithful`), as `docs/PROVER_GUIDE.md` §3 writes it.
- `[rule] Num.MathTotal P .sqrt .binary32`, `[rule] Num.SqrtF32 P` (`tl.sqrt` on `float32`).
- `[rule] Num.DivTotal P .binary32`, `[rule] Num.DivFull P` (`/` on `float32`).

`consistent` takes `Platform.sm90ref` with `exists_total_faithful`.

## Coverage

Each concrete launch of SETUP.md is an instance `setup_* := mk T D BD` (`BT = 16`, `eps = 1e-6`, the
host's `NB`, buffers at `2^30`, `2^31`, `2^31 + 2^30`, a `bfloat16` `y`; `mk T D BD true` for a
`float32` `y`) with a proof `setup_*_ok`:

| SETUP.md launch | `(T, D, BD)` | instance |
|---|---|---|
| host test `(1, 1, 1, 1)` | `(1, 1, 1)` | `setup_1_1_1_1` |
| host test `(1, 1, 1, 128)` | `(1, 128, 128)` | `setup_1_1_1_128` |
| host test `(1, 63, 1, 60)` | `(63, 60, 64)` | `setup_1_63_1_60` |
| host test `(1, 65, 2, 512)` | `(130, 512, 512)` | `setup_1_65_2_512` |
| host test `(2, 500, 4, 64)` | `(4000, 64, 64)` | `setup_2_500_4_64` |
| host test `(2, 1000, 2, 100)` | `(4000, 100, 128)` | `setup_2_1000_2_100` |
| host test `(3, 1024, 4, 128)` | `(12288, 128, 128)` | `setup_3_1024_4_128` |
| host test `(2, 333, 3, 256)` | `(1998, 256, 256)` | `setup_2_333_3_256` |
| host test `(4, 4096, 16, 128)`, DeltaNet's 16 heads of 128 on 4 × 4096 tokens | `(262144, 128, 128)` | `setup_4_4096_16_128` |
| GatedDeltaNet's 6 heads of 256 on 4 × 4096 tokens | `(98304, 256, 256)` | `setup_gdn_4_4096_6_256` |
| PaTH attention's `w`, 32 heads of 64 on 4 × 4096 tokens, `float32` `y` | `(524288, 64, 64)` | `setup_path_4_4096_32_64_f32` |
| PaTH attention's `w`, one decoding token of a batch of 8, `float32` `y` | `(256, 64, 64)` | `setup_path_8_1_32_64_f32` |

The variables SETUP.md names are all inside `Inst.ok`: every `D` in `1 … 512` with its `BD`, every
`T ≥ 1` (up to `16·(2^31 − 1)` rows at `BT = 16`), both output dtypes, `eps = 1e-6` and every
positive `eps` in real use.
The inputs are any finite values whose rows have norm below `2^63`.

## Tolerances

`u`, `η` are one rounding's constants (`TVA.Lemmas.FloatErr`): `float32` `2^-24`, `2^-150`;
`bfloat16` `2^-8`, `2^-134`. `S = Σ_{j<D} a_j²` (exact), `V = S + eps`.

**`rstd`, `2^-12` relative**:
1. Products: each `fl(a_j²)` is within `u·a_j² + η` (`mul_err`); summed, `2^-24·S + 512·2^-150`.
2. Row sum: `tl.sum1c` over the `BD ≤ 512` lanes (masked lanes `+0`), any tree:
   `sumc_err_covers` gives `errBound(3u, 4η, 511, 511, A) = ((1 + 3·2^-24)^511 − 1)·A +
   4·2^-150·511·(1 + 3·2^-24)^511 ≤ 9.14·10^-5·A + 2^-138.99`, `A ≤ (1 + 2^-24)·S + 2^-141`. Its
   room condition (`A + errBound ≤ maxVal/2 − 2η ≈ 2^127`) holds by `Pre`'s `S + eps ≤ 2^126`.
3. `+ eps` (`tl.addc`, `addRel_node`): `3u·(s + eps₃₂) + 4η`; `|eps₃₂ − eps| ≤ 2^-24·eps` (normal).
   Together `|s₂ − V| ≤ 9.2·10^-5·V + 2^-138.6`. With `V ≥ eps ≥ 2^-125` the absolute part is at
   most `2^-13.6·V ≤ 8.1·10^-5·V`, so `|s₂ − V| ≤ 1.73·10^-4·V`. `s₂ ≥ 2^-125 − 2^-149`, a normal number, so
   it is in `sqrtDom`. (Why `2^-125` and not `2^-126`: `AddRel` allows the rounding of `p + q` for any
   `p`, `q` rounding to the operands; with `s₁ = +0`, `p` may be `−2^-150`, and with `eps₃₂ = 2^-126`,
   `q` may be `2^-126 − 2^-150`, whose sum `2^-126 − 2^-149` is subnormal, outside `SqrtF32`'s domain.)
4. `tl.sqrt` (`SqrtF32`): `2^-22·√s₂ + 2^-149`, the absolute part below `2^-86` relative
   (`√s₂ ≥ 2^-63`). `1/√(V(1 + ε))` with `|ε| ≤ 1.73·10^-4` is within `ε/2·(1 + 2ε)` of `1/√V`.
5. `1 / …` (`DivFull`, quotient `≤ 2^63 ≤ 2^126`): `2^-22` relative, `2^-148` absolute, below
   `2^-85` of the quotient (`≥ 2^-64`).

Total `≤ 8.7·10^-5 + 2·2^-22 + … ≤ 8.8·10^-5 < 2^-12 = 2.44·10^-4` (a factor 2.8). The device's error
is far smaller (`2.1·10^-7`, below): the derivation bounds the sum along any tree over the `BD`
lanes (`sumc_err_covers`, depth up to `BD − 1`), a bound linear in the depth, and the absolute
part of step 3 (`2^-138.6` against `V ≥ 2^-125`) alone is `8.1·10^-5`; `2^-12` is within a factor 3
of the derivation's total.

**`y`, `2^-7` relative to `|a_j|·rstdRef`, floor `2^-132`**:
`y = bf16(fl(a_j · rstd))`. With `rstd` within `2^-12` relative, the `float32` product within
`2^-24·|p| + 2^-150`, and the `bfloat16` rounding within `2^-8·|q| + 2^-134` (in range: `|q| ≤ 1.001`),
the error is at most `((1 + 2^-12)(1 + 2^-24)(1 + 2^-8) − 1)·|a_j|·rstdRef + 2^-134 + 2^-150·(1 + 2^-8)`
`≤ 4.2·10^-3·|a_j|·rstdRef + 2^-133.9`, inside `2^-7 = 7.8·10^-3` and `2^-132`. The relative part is
twice the `bfloat16` rounding the kernel cannot avoid (`2^-8`), and the GPU reaches `0.498` of it. The
floor covers subnormal outputs (`|y| < 2^-126`, e.g. `eps = 2^100`, `a = 2^-80`).

**A `float32` `y`, `2^-11` relative to `|a_j|·rstdRef`, floor `2^-149`**:
`y = fl(a_j · rstd)`, stored as it is (`.to(float32)` of a `float32` value is the value). With
`rstd` within `2^-12` relative and the `float32` product within `2^-24·|p| + 2^-150`, the error is at
most `((1 + 2^-12)(1 + 2^-24) − 1)·|a_j|·rstdRef + 2^-150 ≤ 2.45·10^-4·|a_j|·rstdRef + 2^-150`, inside
`2^-11 = 4.88·10^-4` and `2^-149`. The relative part is twice `rstd`'s tolerance, which it inherits;
the GPU reaches `0.34` of it (on an output near `2^-150`, where the floor dominates) and `5·10^-4`
on the real shapes.

## Exclusions

- **`eps < 2^-125`.** A limit of the kernel below `2^-126`: when `eps` rounds to a `float32` below
  `2^-126`, a zero row's `sum + eps` is subnormal or zero, `sqrt.approx.ftz.f32` flushes it to `0`, and
  the kernel stores `rstd = +inf` and `y = 0·inf = NaN` (reference `rstd = eps^-1/2`, `y = 0`). First
  launch: `T = 1`, `D = 4`, `x = 0`, `eps = 1e-39` (checked on the GPU: `rstd inf`, `y nan`). In
  `[2^-126, 2^-125)` the kernel is correct: this part of the exclusion is a **library gap, not a kernel
  limit** (the library's fusable addition, `AddRel`, admits an operand below the `float32` `eps` and so a
  subnormal sum, Tolerances, step 3), so the claim starts at `2^-125`. Real use does not reach either: `eps` is
  `1e-6` by default, and no caller uses one below `1e-12`.
- **Rows with `Σ x² + eps > 2^126`, and `eps > 2^126`.** A limit of the kernel near `2^128`: the
  `float32` sum of squares overflows to `+inf`, and the kernel stores `rstd = 0`, `y = 0` (reference
  `y = ±1` for one element). First launch: `T = 1`, `D = 1`, `x = 2^64` (checked: `rstd 0`, `y 0`).
  Between `2^126` and the overflow (`eps ∈ (2^126, 2^128)` with small rows, and rows whose `Σ x² + eps`
  lies there) the kernel is correct: this part is a **library gap, not a kernel limit** (the library's
  sum bound needs room below `maxVal/2`). Real use (normalised queries and keys) has row norms many
  orders of magnitude below `2^63`.
- **Non-finite inputs** (`±inf`, NaN in `x`, or a non-finite `eps`): the reference is undefined; the
  kernel propagates NaN. Not a correctness question.
- **`cdiv(T, BT) ≥ 2^31`**: CUDA rejects the grid at launch (no outcome); at `BT = 16`, more than
  `3.4·10^10` rows, never reached (at `D = 64` that is 4 TiB of `x`).
- **An `output_dtype` other than `None` and `torch.float32`**, and a `float32` or `float16` `x`: not
  part of the setup (fla's callers pass `bfloat16` activations, and only PaTH attention asks for
  another output dtype, `float32`).
- **`D > 512`, `D = 0`**: not this kernel's path (the host asserts `D ≤ 512`; fla launches
  `l2norm_fwd_kernel_row` beyond); an empty row has no view.

## Library requests

None needed for this claim. The library's `AddRel` takes one operand exact (`AddRelWide` is the wider relation); with it the `[2^-126, 2^-125)` range of `eps` left out above could be covered.

## Checks

1. **Builds**: `lake env lean ../claims/l2norm_fwd_bf16/Claim.lean` from `lean/`: no errors, no
   warnings, no `sorry`, no `axiom`; `consistent`, `satisfiable` and every `setup_*_ok` depend on
   `propext`, `Classical.choice`, `Quot.sound` only.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (`setup_1_1_1_1`, a zero memory).
3. **Covers the setup**: the twelve `setup_*_ok` above.
4. **GPU** (`check_gpu.py`, H200, Triton 3.8.0): 230 launches, all within `Post`: the twelve SETUP.md
   launches; `D` in 17 classes from 1 to 512 at `T` 1, 15, 17, 1001 (and at `T = 17` with a
   `float32` `y`); `eps` from `2^-125` to `2^100` (and `2^126` on zero rows) with zero rows, subnormal
   inputs, squares below `float32`'s range and rows scaled by `2^±40`; rows with `Σ x² ` just below
   `2^126` and mixed scales (`2^60` beside `2^-100`); direct launches at `BT` 1, 2, 8, 16, 32, 64, 128
   with the outputs inside sentinel-filled buffers (the frame); and `T = 2^31 + 7` (the `int64`
   class) on the first two programs' rows; every edge with both output dtypes. Worst error /
   tolerance: `rstd` `8.8·10^-4` (margin `1136×`), `bfloat16` `y` `0.498` (margin `2.0×`),
   `float32` `y` `0.34` (margin `2.9×`); every output finite; `x` unchanged and every sentinel
   intact. Wrong outputs fail `Post` with both dtypes: `y = 0`, `y = x`, `y` with the next row's
   `rstd`, `rstd·(1 + 2^-11)`, `rstd = 1/Σ x²`, and a `float32` `y` rounded to `bfloat16` (ratio `8`).
   `tol / |y|` at the real shapes: `7.8·10^-3` (`bfloat16`), `4.9·10^-4` (`float32`). The two
   exclusions' first launches give `rstd = inf, y = NaN` and `rstd = 0, y = 0`.
