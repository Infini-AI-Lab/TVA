# `l2norm_fwd_bf16`

**The kernel.** `l2norm_fwd_kernel` of fla (`source/l2norm.py`, kernel lines 44–60; host
`l2norm_fwd` lines 63–90, which launches it with `BT = 16`, `num_warps = 4` on the grid
`(cdiv(T, 16),)`). Program `p` takes the rows `p·BT + [0, BT)` of `x` viewed as `[T, D]` (`int64`
row offsets, masked by `T`) and the columns `[0, BD)` (masked by `D`), loads the tile with
`other = 0`, widens it to `float32`, sums the squares of each row, adds `eps`, takes `tl.sqrt` and
`1 / sqrt`, stores `rstd`, and stores `x·rstd` to `y` (rounded to `bfloat16`, or kept `float32`).
The claim is `Claims.L2NormFwd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for each
row `t < T`, with `a_j` the row's initial `bfloat16` inputs and `rstdRef = 1 / sqrt(Σ_{j<D} a_j² +
eps)` exact in the reals with the host's double `eps`:
- `rstd[t]` is a finite `float32` with `|rstd[t] − rstdRef| ≤ 2^-12 · rstdRef`;
- each `y[t, j]`, `j < D`, is a finite value of `y`'s dtype with
  `|y[t, j] − a_j·rstdRef| ≤ 2^-7 · |a_j|·rstdRef + 2^-132` for a `bfloat16` `y`
  (`output_dtype = None`), and `≤ 2^-11 · |a_j|·rstdRef + 2^-149` for a `float32` `y`
  (`output_dtype = torch.float32`);
- every byte outside the first `T·D` elements of `y` and the first `T` elements of `rstd` is
  unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ D ≤ 512` (this kernel's path; the host asserts `D ≤ 512`); `BD` a power of two with
  `D ≤ BD ≤ 512` (the host's `next_power_of_2(D)`);
- `BT` any power of two with `BT·BD ≤ 2^20` (the host's `16`, and every `BT` of fla's autotune
  list);
- fewer than `2^31` programs (`cdiv(T, BT) < 2^31`); `T = 0` is covered. `T` is passed as an
  `int32` below `2^31` and as an `int64` from `2^31`; both are covered;
- `y` is `bfloat16` or `float32`;
- `x` and `y` hold at least `T·D` elements, `rstd` at least `T`, all valid in the initial memory;
  the written elements of `y` and `rstd` share no byte with each other or with the read elements of
  `x` (the host allocates both fresh);
- `eps` is a finite double in `[2^-125, 2^126]` (the default is `1e-6`);
- every element read is a finite `bfloat16`, and every row has `Σ_j x[t, j]² + eps ≤ 2^126`.

**The launches covered** (`SETUP.md`): the host's eight test shapes `(B, T, H, D)`, DeltaNet's 16
heads of 128 and GatedDeltaNet's 6 heads of 256 on 4 × 4096 tokens, and PaTH attention's
`float32` `w` (32 heads of 64, on 4 × 4096 tokens and on one decoding token of a batch of 8)
(`setup_*_ok`). `mk_ok` covers the host's launch for every `D ≤ 512` with a power-of-two
`BD ≥ D`, with `T ≤ 2^27` and `T·D ≤ 2^28`, and either output dtype.

**Device facts** the tolerances rest on (all in `Device P`): `Device.SqrtF32` (`tl.sqrt` on
`float32`: `2^-22·√x + 2^-149` on `sqrtDom`), `Device.DivFull` (`/` on `float32`: relative
`2^-22`, absolute `2^-148`, for quotients up to `2^126`), and the totality facts for `sqrt` and `/`.

## Tolerances

With `u = 2^-24`, `η = 2^-150` (one `float32` rounding), `S = Σ_j a_j²` and `V = S + eps`:
- `rstd`, `2^-12` relative: the products and the fusable row sum over at most `512` lanes, along
  any tree, are within `9.14·10^-5·S + 2^-138.99`; the fusable `+ eps` brings `s₂` within
  `1.73·10^-4·V` of `V`, a normal number (so in `tl.sqrt`'s domain, from `eps ≥ 2^-125`);
  `tl.sqrt` and `/` add `2^-22` each. Total below `8.8·10^-5 < 2^-12`.
- a `bfloat16` `y`, `2^-7` relative, floor `2^-132`: `((1 + 2^-12)(1 + 2^-24)(1 + 2^-8) − 1) ≤
  4.2·10^-3` relative and `2^-133.9` absolute. The relative part is twice the `bfloat16` rounding
  the kernel cannot avoid.
- a `float32` `y`, `2^-11` relative, floor `2^-149`: `((1 + 2^-12)(1 + 2^-24) − 1) ≤ 2.45·10^-4`
  relative and `2^-150` absolute.

## Exclusions

- **`eps < 2^-125`.** Below `2^-126` the kernel is wrong: a zero row's `sum + eps` is subnormal,
  `tl.sqrt` flushes it to `0`, and the kernel stores `rstd = +inf`, `y = NaN` (`T = 1`, `D = 4`,
  `x = 0`, `eps = 1e-39`). In `[2^-126, 2^-125)` the kernel is correct; the claim leaves it out
  because the fusable addition `AddRel` admits an operand below the `float32` `eps`, and so a
  subnormal sum. Real use has `eps ≥ 1e-12`.
- **Rows with `Σ x² + eps > 2^126`, and `eps > 2^126`.** Near `2^128` the `float32` sum of squares
  overflows and the kernel stores `rstd = 0`, `y = 0` (`T = 1`, `D = 1`, `x = 2^64`); between
  `2^126` and the overflow the kernel is correct, but the sum's error bound needs room below
  `maxVal/2`. Real row norms are far below `2^63`.
- **Non-finite inputs** (in `x`, or `eps`): the reference is undefined.
- **`cdiv(T, BT) ≥ 2^31`**: CUDA rejects the grid at launch.
- **Other dtypes** of `x` or `y`, **`D > 512`** (fla launches another kernel) and **`D = 0`**:
  outside the file's launches.

`check_gpu.py` (one H200) checks `Post` on the `SETUP.md` launches and on the edges of `Inst.ok`,
with both output dtypes, and runs the first launches of the exclusions.
