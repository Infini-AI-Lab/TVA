# `layer_norm_fwd_f16`

**The kernel.** `_layer_norm_fwd_fused` of Triton 3.8.0's tutorial 05, "Layer Normalization"
(`source/05-layer-norm.py`, kernel lines 48–95; host `LayerNorm.forward` lines 235–260). One
program per row (grid `(M,)`): it sums the row in a block of `BLOCK_SIZE = next_power_of_2(N)`
lanes to get the mean, sums the squared deviations to get the variance, stores `mean` and
`rstd = 1/sqrt(var + eps)` (`float32`), and writes `y = (x − mean)·rstd·w + b` as `float16`. The
claim is `Claims.LayerNormFwdF16.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
row `t < M`, with the exact real LayerNorm of the row as reference (`meanRef = Σ x/N`,
`varRef = Σ (x − meanRef)²/N`, `rstdRef = 1/sqrt(varRef + eps)` with the host's double `eps`,
`zRef j = (x j − meanRef)·rstdRef`, `yRef j = zRef j·w j + b j`):
- `mean[t]` is a finite `float32` within `meanTol` of `meanRef`;
- `rstd[t]` is a finite `float32` within `rstdTol` of `rstdRef`;
- `y[t, j]` is a finite `float16` within `yTol j` of `yRef j`;
- every byte outside the elements of `y`, `mean` and `rstd` is unchanged.

With `D = log₂ BLOCK_SIZE`, `A = Σ|x|/N` and the row's condition number `cond = meanTol·rstdRef`:

| Output | Tolerance |
|---|---|
| `mean` | `meanTol = (D + 4)·2^-23·A + 2^-140` |
| `rstd` | `cond ≤ 1/2`: `(D·2^-21 + 2^-140/eps + cond/4)·rstdRef + 2^-140`; `cond > 1/2`: `2/sqrt(eps)` |
| `y` | `yTol = 2^-9·(|w|·|zRef| + |b|) + 2·|w|·zTol + 2^-24` |

`zTol`, the bound on the kernel's standardised value, is
`(1 + 2·gam)·(|zRef|·(gam + mTerm) + cond) + 2^-140·(1 + rstdRef)` when `cond ≤ 1/2`, and
`|zRef| + (1 + 2·gam)·(|x − meanRef| + meanTol)/sqrt(eps) + 2^-140·(1 + 1/sqrt(eps))` otherwise,
with `gam = (D + 8)·(2^-23 + 2^-145/eps)` and `mTerm = cond/8 + cond²/4`. The tolerances hold on
every row, with no condition on the data's spread.

## Conditions

`Inst.ok` and `Pre`:
- `2 ≤ N ≤ 32768`, `N` a multiple of `128`; `BLOCK_SIZE` is a power of two with
  `N ≤ BLOCK_SIZE < 2N` (`next_power_of_2(N)`);
- `1 ≤ M ≤ 2^22` and `(M − 1)·N < 2^31` (line 63's `int32` product `row * stride`);
- `eps` is finite, `2^-125 ≤ eps ≤ 2^126`;
- each tensor holds what the launch touches; the outputs share no byte with the inputs or with each
  other (fresh `torch.empty` tensors);
- the six buffers are valid in the initial memory; every element of `x`, `weight`, `bias` the
  launch reads is finite; every row's output has room in `float16` (`YRoom`:
  `|w j|·(|zRef j| + zTol j) + |b j| ≤ 2^15` for every column).

**The launches covered** (`SETUP.md`): the test (`M = 1151`, `N = 8192`, `setup_test_ok`) and the
benchmark (`M = 4096`, `N = 512·i`, `2 ≤ i ≤ 31`, `setup_bench_ok`), with `eps = 1e-5`. `mkInst_ok`
covers the host's launch for every `N` a multiple of `128` up to `32768`, `1 ≤ M ≤ 2^22`,
`M·N ≤ 2^30`.

**Device facts** the tolerances rest on (in `Device P`): `Device.SqrtF32` (`tl.sqrt` on `float32`:
relative `2^-22`, absolute `2^-149`, on `0` and inputs `≥ 2^-126`), `Device.DivFull` (`/` on
`float32`: relative `2^-22`, absolute `2^-148`), `Device.SumDepth` (the tree of a float `tl.sum`
over `B` positions has depth at most `⌈log₂ B⌉`), and the totality of `tl.sqrt` and `/`.

## Tolerances

Derived step by step, with `u = 2^-24`, `η = 2^-150` and the fused `x − mean` and `var + eps` taken
by the library's `AddRel` (`TVA.Triton.Arith`):
1. **Mean**: a `tl.sum` of depth `D` and one division give `(E0 + 4u(1 + E0))·A` plus less than `2^-140`,
   with `E0 = (1 + u)^D − 1`: at most `0.501` of `meanTol`'s relative part.
2. **Variance**: each deviation is `(1 ± u)(x − meanRef)` minus the mean's error; with
   `Σ (x − meanRef) = 0` and Cauchy–Schwarz, the input of `tl.sqrt` is `(S₀/N + eps)(1 + Θ)` with
   `Θ` a product of `(1 ± u)`-factors, the sum's `(1 ± E1)` and the division's `(1 ± 4u)`.
3. **rstd**: `tl.sqrt` and `1/·` add `(1 ± 4u)` each; for `cond ≤ 1/2` the mean's error moves
   `rstd` by at most `mTerm` relatively, which gives `rstdRef·(gam + mTerm) + 2^-140`; for
   `cond > 1/2` both `rstd` and `rstdRef` are below `(1 + gam)/sqrt(eps) + 2^-140`.
4. **Standardised value**: line 92 recomputes the deviation; the error is `zTol`.
5. **Output**: `mul_add(z, w, b)` then one `float16` rounding (no overflow, by `YRoom`).
6. Relaxed to the stated tolerances using `N ≥ 128` (`7 ≤ D ≤ 15`) and `eps ≥ 2^-125`.

A kernel that drops one term of the mean's sum or of the variance's sum, or divides the variance
by `N − 1`, fails `Post` on the file's rows.

## Exclusions

- **`N` not a multiple of `128` or above `2^17`, `M > 2^22`**: the claim covers real model shapes
  only. Every launch of the file is covered.
- **`(M − 1)·N ≥ 2^31`**: a limit of the kernel; `row * stride` wraps in `int32`.
- **`eps < 2^-125`**: below `2^-126` a limit of the kernel (a subnormal `var + eps` is flushed by
  `tl.sqrt`, giving `rstd = inf`, `y = NaN`); in `[2^-126, 2^-125)` the semantics do not exclude a
  subnormal input of `tl.sqrt`. Real `eps` (`1e-5` to `1e-12`) reaches neither.
- **`eps > 2^126`**: outside real use.
- **Outputs without room in `float16`** (`YRoom` false): a limit of the output dtype; on rows of
  `cond > 1/2` it can also fail where the device stays finite, because the semantics let the two
  fused subtractions round differently. Outside real use.
- **Non-finite inputs**: no reference value.
- **`N = 1`**: a different compiled kernel (`N` and `stride` become compile-time constants); not
  part of the setup.
- **`N > 32768`**: the host raises (no launch).
- **The backward kernels, other dtypes**: not part of the setup.

`check_gpu.py` (one H200) checks `Post` on the file's launches, edge cases and wrong kernels.
