import TVA

/-!
# `layer_norm_fwd_f16`: the prepared claim

Source: `source/05-layer-norm.py` (Triton 3.8.0's tutorial 05, unchanged). The kernel is
`_layer_norm_fwd_fused` (lines 48–95); it reads the module-level names `triton` and `tl`, bound by
the imports of lines 34–35. Its host `LayerNorm.forward` (lines 235–260) launches it with one
program per row, the grid `(M,)`, on a `float16` input `x` (`[M, N]`, contiguous), `float16`
`weight` and `bias` (`[N]`), writing the `float16` output `y = torch.empty_like(x)` and the
`float32` `mean` and `rstd` (`[M]`), with `BLOCK_SIZE = min(32768, next_power_of_2(N))` (the host
raises when `N > BLOCK_SIZE`), `num_warps = min(max(BLOCK_SIZE // 256, 1), 8)` and `num_ctas = 1`.

* The model: `layer_norm_fwd_fused`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`. Per row: the stored mean, `rstd` and every output element are
  within tolerances of the exact real LayerNorm of the row. The tolerances hold on every row: they
  grow with the row's condition number `cond = meanTol · rstdRef`, which `eps` bounds.
* The assumptions: `Assumptions` (`Compiler.Faithful`, the `tl.sqrt` and `/` facts, and the depth
  of a `tl.sum`'s tree).
* The claim: `Statement`.
* Checked here: `consistent`, `satisfiable`, and the coverage facts `mkInst_ok`, `setup_*_ok`.
-/

namespace Claims.LayerNormFwdF16

open TVA TVA.Math.Float

/-! ## The model -/

/-- `other=0.` of the loads of `X` (lines 70, 77, 91): the Python `float` `0.0` becomes a `float32`
tensor, then the pointer's dtype `float16` (SEMANTICS.md, `tl.load`). -/
def other16 (BLOCK_SIZE : Nat) : Prog (Tensor [BLOCK_SIZE] F16) :=
  tl.convert .f32 .f16 (tl.full [BLOCK_SIZE] (FloatV.litF32 (FloatV.ofPat 0)))

set_option linter.unusedVariables false in
/-- `_layer_norm_fwd_fused` (source lines 48–95). The pointers are `*fp16` (`X`, `Y`, `W`, `B`) and
`*fp32` (`Mean`, `Rstd`), the dtypes of the covered launches. `stride` and `N` are runtime `int32`
scalars (Python `int` arguments other than `1`: `Inst.ok` keeps `N ≥ 2`, and `stride = N`); `eps`
is the runtime `float32` scalar (a Python `float` argument, `FloatV.hostF32` of the double, see
`model`); `BLOCK_SIZE` is the `tl.constexpr`, a Lean parameter. -/
def layer_norm_fwd_fused (X Y W B : Ptr .f16) (Mean Rstd : Ptr .f32) (stride N : I32) (eps : F32)
    (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 62
    let row := tl.program_id ctx 0
    -- lines 63–64: `Y += row * stride`, `X += row * stride`: the `int32` product (wrapping), the
    -- pointers advanced by it
    let Y := Y + row * stride
    let X := X + row * stride
    -- line 66: `mean = 0` binds a Python number that line 72 replaces before any use: no step
    -- line 67
    let _mean := tl.zeros [BLOCK_SIZE] .f32
    -- line 68: `range(0, N, BLOCK_SIZE)` with a runtime bound: the `int32` loop (the Python `0` an
    -- `int32`, the constexpr step built in `int32`), carrying `_mean`
    let _mean ← tl.forRange (IntV.ofInt 0 : I32) N (IntV.ofNat BLOCK_SIZE : I32) _mean
        fun off _mean => do
      -- line 69: the `int32` scalar plus the `int32` block
      let cols := off + tl.arange 0 BLOCK_SIZE
      -- line 70: the `float16` block, `other=0.`, converted to `float32` exactly
      let a : Tensor [BLOCK_SIZE] F32 ← tl.cvt (← tl.load (X + cols) (some (cols.ltS N))
        (some (← other16 BLOCK_SIZE)))
      -- line 71: `_mean += a`: the IEEE `+` (no product feeds it)
      tl.add _mean a
    -- line 72: `tl.sum(_mean, axis=0)`: the reduction of a block that is not a product (IEEE
    -- additions); `/ N`: the `float32` scalar by the `int32` scalar `N`, which promotion converts
    -- to `float32` (`tl.itof`), then the platform's division
    let nf1 : Tensor [] F32 ← tl.itof (Tensor.scalar N)
    let mean := (← tl.div (Tensor.scalar (← tl.sum _mean)) nf1).item
    -- line 74
    let _var := tl.zeros [BLOCK_SIZE] .f32
    -- line 75: as line 68, carrying `_var`
    let _var ← tl.forRange (IntV.ofInt 0 : I32) N (IntV.ofNat BLOCK_SIZE : I32) _var
        fun off _var => do
      -- line 76
      let cols := off + tl.arange 0 BLOCK_SIZE
      -- line 77: as line 70
      let x : Tensor [BLOCK_SIZE] F32 ← tl.cvt (← tl.load (X + cols) (some (cols.ltS N))
        (some (← other16 BLOCK_SIZE)))
      -- line 78: `x - mean`: `mean` is a `float32` quotient (line 72), a product for the fusion
      -- rule, so the fusable `-` (`tl.subc`), the scalar broadcast; the lanes `cols ≥ N` hold the
      -- constant `0.`, built in `float32`
      let x := tl.where (cols.ltS N) (← tl.subc x (bc mean)) (bc (FloatV.litF32 (FloatV.ofPat 0)))
      -- line 79: `_var += x * x`: an addition the product feeds (`tl.mul_add`)
      tl.mul_add x x _var
    -- line 80: as line 72 (`_var` holds `tl.mul_add` results, not products: IEEE additions)
    let nf2 : Tensor [] F32 ← tl.itof (Tensor.scalar N)
    let var := (← tl.div (Tensor.scalar (← tl.sum _var)) nf2).item
    -- line 81: `var + eps`: `var` is a `float32` quotient (line 80), a product for the fusion
    -- rule, so the fusable `+` (`tl.addc`); `tl.sqrt` the platform's square root; `1 / ·`: the
    -- Python `1` built in `float32`, the platform's division
    let rstd := (← tl.div (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
      (← tl.sqrt (← tl.addc (Tensor.scalar var) (Tensor.scalar eps)))).item
    -- line 83: `Mean + row`, the `float32` scalar (the pointer's dtype)
    tl.store1 (Mean + row) mean
    -- line 84
    tl.store1 (Rstd + row) rstd
    -- line 86: as line 68, no carried name (every name the body assigns is first defined there
    -- and not used after the loop)
    tl.forRange (IntV.ofInt 0 : I32) N (IntV.ofNat BLOCK_SIZE : I32) () fun off _ => do
      -- line 87
      let cols := off + tl.arange 0 BLOCK_SIZE
      -- line 88
      let mask := cols.ltS N
      -- lines 89–90: no `other`: the lanes `cols ≥ N` hold unspecified values, which no stored
      -- lane uses
      let w ← tl.load (W + cols) (some mask)
      let b ← tl.load (B + cols) (some mask)
      -- line 91: as line 70
      let x : Tensor [BLOCK_SIZE] F32 ← tl.cvt (← tl.load (X + cols) (some mask)
        (some (← other16 BLOCK_SIZE)))
      -- line 92: `(x - mean) * rstd`: the fusable `-` on the quotient `mean` (`tl.subc`, as line
      -- 78), then the IEEE `*` (it feeds a product)
      let x_hat ← tl.mul (← tl.subc x (bc mean)) (bc rstd)
      -- line 93: `x_hat * w + b`: `w` and `b` promoted to `float32` (exactly), the addition the
      -- product feeds (`tl.mul_add`)
      let w32 : Tensor [BLOCK_SIZE] F32 ← tl.cvt w
      let b32 : Tensor [BLOCK_SIZE] F32 ← tl.cvt b
      let y ← tl.mul_add x_hat w32 b32
      -- line 95: the `float32` block stored through the `*fp16` pointer: converted to `float16`,
      -- one rounding to nearest even
      let y16 : Tensor [BLOCK_SIZE] F16 ← tl.cvt y
      tl.store (Y + cols) y16 (some mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of `_layer_norm_fwd_fused` through `layer_norm(x, w_shape,
weight, bias, eps)` (`LayerNorm.forward`, lines 235–260) on `float16` tensors, given by their
storage (`x` and `y` contiguous `[M, N]`, `weight` and `bias` `[N]`, `mean` and `rstd` `[M]`). -/
structure Inst where
  /-- `M` (the rows of `x.reshape(-1, x.shape[-1])`). -/
  M : Nat
  /-- `N` (`x.shape[-1]`): the row length; also `x_arg.stride(0)` (contiguous rows). -/
  N : Nat
  /-- `BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(N))` (line 246). -/
  BLOCK : Nat
  /-- `eps`, the Python `float` (a double; the test and the benchmark pass `1e-5`). -/
  eps : F64
  /-- `x`, `float16`. -/
  x : Buf .f16
  /-- `weight`, `float16`. -/
  w : Buf .f16
  /-- `bias`, `float16`. -/
  b : Buf .f16
  /-- `y` (`torch.empty_like(x)`, line 238), `float16`. -/
  y : Buf .f16
  /-- `mean` (`torch.empty((M,), dtype=torch.float32)`, line 242). -/
  mean : Buf .f32
  /-- `rstd` (line 243). -/
  rstd : Buf .f32

/-- Element `(t, j)` of a contiguous `[M, N]` tensor. -/
def idx (d : Inst) (t j : Nat) : Nat := t * d.N + j

/-- The elements of an `[M, N]` tensor. -/
def matSet (d : Inst) : Set Nat := {k | k < d.M * d.N}

/-- The elements of an `[M]` tensor. -/
def rowSet (d : Inst) : Set Nat := {k | k < d.M}

/-- The elements of an `[N]` tensor. -/
def vecSet (d : Inst) : Set Nat := {k | k < d.N}

/-- The bytes the launch reads. -/
def inBytes (d : Inst) : Set ℤ :=
  d.x.elems (matSet d) ∪ d.w.elems (vecSet d) ∪ d.b.elems (vecSet d)

/-- The bytes the launch writes. -/
def outBytes (d : Inst) : Set ℤ :=
  d.y.elems (matSet d) ∪ d.mean.elems (rowSet d) ∪ d.rstd.elems (rowSet d)

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `N ≥ 2`: the host passes `N` and `stride = N` as Python `int`s, runtime `int32` scalars
  unless equal to `1`, which Triton makes a compile-time constant (README, Exclusions). -/
  n_ge : 2 ≤ d.N
  /-- `N ≤ 32768`: the host raises for `N > BLOCK_SIZE` (line 247), which is `N > 32768` for
  `float16`. -/
  n_le : d.N ≤ 32768
  /-- `BLOCK_SIZE` is a power of two. -/
  block_pow2 : ∃ e, d.BLOCK = 2 ^ e
  /-- `N ≤ BLOCK_SIZE`. -/
  n_le_block : d.N ≤ d.BLOCK
  /-- `BLOCK_SIZE < 2·N`: with `block_pow2` and `n_le_block`, `BLOCK_SIZE = next_power_of_2(N)`
  (line 246; below `32768` it is not capped). -/
  block_lt : d.BLOCK < 2 * d.N
  /-- At least one row. -/
  m_pos : 1 ≤ d.M
  /-- `(M − 1)·N < 2^31`: line 63's `row * stride` is an `int32` product, which wraps beyond
  (README, Exclusions: a limit of the kernel); it also keeps the grid `M` below `2^31`. -/
  rows_lt : (d.M - 1) * d.N < 2 ^ 31
  /-- `N` is a model dimension (the hidden size, the row length): a multiple of `128`, at most
  `2^17` (README, Exclusions). -/
  n_dim : d.N % 128 = 0 ∧ d.N ≤ 2 ^ 17
  /-- `M` is a token count (one row per token of the batch): at most `2^22` (README, Exclusions). -/
  m_ctx : d.M ≤ 2 ^ 22
  /-- `eps` is finite. -/
  eps_fin : d.eps.isFinite = true
  /-- `eps ≥ 2^-125` (README, Exclusions). -/
  eps_ge : (2 : ℚ) ^ (-125 : ℤ) ≤ d.eps.toRat
  /-- `eps ≤ 2^126` (README, Exclusions). -/
  eps_le : d.eps.toRat ≤ (2 : ℚ) ^ (126 : ℤ)
  /-- `x` holds `M·N` elements. -/
  x_len : d.M * d.N ≤ d.x.len
  /-- `weight` holds `N` elements. -/
  w_len : d.N ≤ d.w.len
  /-- `bias` holds `N` elements. -/
  b_len : d.N ≤ d.b.len
  /-- `y` holds `M·N` elements. -/
  y_len : d.M * d.N ≤ d.y.len
  /-- `mean` holds `M` elements. -/
  mean_len : d.M ≤ d.mean.len
  /-- `rstd` holds `M` elements. -/
  rstd_len : d.M ≤ d.rstd.len
  /-- The outputs share no byte with the inputs (`torch.empty` allocates them). -/
  out_in : Disjoint (outBytes d) (inBytes d)
  /-- `y` and `mean` share no byte. -/
  y_mean : Disjoint (d.y.elems (matSet d)) (d.mean.elems (rowSet d))
  /-- `y` and `rstd` share no byte. -/
  y_rstd : Disjoint (d.y.elems (matSet d)) (d.rstd.elems (rowSet d))
  /-- `mean` and `rstd` share no byte. -/
  mean_rstd : Disjoint (d.mean.elems (rowSet d)) (d.rstd.elems (rowSet d))

/-- The model on an instance: the buffers' first elements, `stride = N` and `N` as `int32`, `eps`
as the launcher's `float32` (`FloatV.hostF32` of the Python double), and the constexpr. -/
def model (d : Inst) : Kernel :=
  layer_norm_fwd_fused (d.x.ptr 0) (d.y.ptr 0) (d.w.ptr 0) (d.b.ptr 0) (d.mean.ptr 0)
    (d.rstd.ptr 0) (IntV.ofNat d.N) (IntV.ofNat d.N) (FloatV.hostF32 d.eps) d.BLOCK

/-- The host's launch (lines 252–255): the grid `(M,)`, the source's block shape `[BLOCK_SIZE]`
(`cols`, `mask`, the loads, the arithmetic, the store); `num_ctas = 1` (`grid1`'s), no `launch_pdl`,
`debug = false`, `enable_fp_fusion` Triton's default (`true`); `num_warps` is performance-only. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 d.M [[d.BLOCK]]

/-- The `float16` element `i` of a buffer in memory `m` (`+0` where absent; `Pre` makes every element
the launch reads present). -/
def rval (m : Mem) (p : Buf .f16) (i : Nat) : F16 :=
  match m.read .f16 (p.ptr (i : Int)).addr with
  | some v => v
  | none => FloatV.ofPat 0

/-- `x[t, j]`, `weight[j]`, `bias[j]` in the initial memory. -/
def xv (d : Inst) (init : Mem) (t j : Nat) : F16 := rval init d.x (idx d t j)
def wv (d : Inst) (init : Mem) (j : Nat) : F16 := rval init d.w j
def bv (d : Inst) (init : Mem) (j : Nat) : F16 := rval init d.b j

/-! ## The specification

For row `t`, from the host's inputs (`x[t, j]` as a real, exactly), the reference is the exact real
LayerNorm: `meanRef = Σ x / N`, `varRef = Σ (x − meanRef)² / N`, `rstdRef = 1 / sqrt(varRef + eps)`
(the Python double `eps`), `zRef j = (x j − meanRef)·rstdRef` and `yRef j = zRef j · w j + b j`.
The tolerances (README, Tolerances, for their derivation), with `D = log₂ BLOCK_SIZE` (the depth of
the reductions' trees over their `BLOCK_SIZE` leaves, `Rules.Num.SumDepth`):

* `meanTol = (D + 4)·2^-23 · Σ|x|/N + 2^-140`;
* `cond = meanTol · rstdRef`, the row's condition number;
* `rstdTol = (D·2^-21 + 2^-140/eps + cond/4)·rstdRef + 2^-140` when `cond ≤ 1/2`, and
  `2/sqrt(eps)` when `cond > 1/2`;
* `zTol j`, the bound on the kernel's standardised value (`Pre`'s `YRoom` uses it too), with
  `gam = (D + 8)·(2^-23 + 2^-145 / eps)` and `mTerm = cond/8 + cond²/4`:
  `(1 + 2·gam)·(|zRef j|·(gam + mTerm) + cond) + 2^-140·(1 + rstdRef)` when `cond ≤ 1/2`, and
  `|zRef j| + (1 + 2·gam)·(|x j − meanRef| + meanTol)/sqrt(eps) + 2^-140·(1 + 1/sqrt(eps))` when
  `cond > 1/2`;
* `yTol j = 2^-9·(|w j|·|zRef j| + |b j|) + 2·|w j|·zTol j + 2^-24`.
-/

/-- `x[t, j]` as a real. -/
noncomputable def xR (d : Inst) (init : Mem) (t j : Nat) : ℝ := (xv d init t j).toReal

/-- The row's exact mean `Σ_j x j / N`. -/
noncomputable def meanRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.N, xR d init t j) / d.N

/-- The row's exact mean magnitude `Σ_j |x j| / N`. -/
noncomputable def absMean (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.N, |xR d init t j|) / d.N

/-- The row's exact variance `Σ_j (x j − meanRef)² / N`. -/
noncomputable def varRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.N, (xR d init t j - meanRef d init t) ^ 2) / d.N

/-- The host's `eps` (the double) as a real. -/
noncomputable def epsR (d : Inst) : ℝ := d.eps.toReal

/-- The exact `1 / sqrt(varRef + eps)`. -/
noncomputable def rstdRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  1 / Real.sqrt (varRef d init t + epsR d)

/-- The exact standardised value `(x j − meanRef)·rstdRef`. -/
noncomputable def zRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  (xR d init t j - meanRef d init t) * rstdRef d init t

/-- The exact output `zRef j · w j + b j`. -/
noncomputable def yRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  zRef d init t j * (wv d init j).toReal + (bv d init j).toReal

/-- `D = ⌈log₂ BLOCK_SIZE⌉` (`log₂ BLOCK_SIZE`, a power of two): the largest depth of the trees of
the two `tl.sum`s over their `BLOCK_SIZE` positions on sm_90 (`Rules.Num.SumDepth`). -/
def depth (d : Inst) : Nat := Nat.clog 2 d.BLOCK

/-- The tolerance of the stored mean. -/
noncomputable def meanTol (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  ((depth d : ℝ) + 4) * 2 ^ (-23 : ℤ) * absMean d init t + 2 ^ (-140 : ℤ)

/-- The row's condition number: the mean's tolerance in units of the row's standard deviation
(with `eps`). At most `meanTol / sqrt(eps)`. -/
noncomputable def cond (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  meanTol d init t * rstdRef d init t

/-- The relative error of `rstd` from the roundings of the variance, `eps`, `tl.sqrt` and `/`
(in `zTol`). -/
noncomputable def gam (d : Inst) : ℝ :=
  ((depth d : ℝ) + 8) * (2 ^ (-23 : ℤ) + 2 ^ (-145 : ℤ) / epsR d)

/-- The relative error of `rstd` that the mean's error causes, on a row with `cond ≤ 1/2` (in
`zTol`): the
deviations of lines 78 and 92 subtract, lane by lane, a value within `meanTol` of `meanRef` (the
fused subtraction takes an exact value that rounds to the stored mean). -/
noncomputable def mTerm (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  cond d init t / 8 + cond d init t ^ 2 / 4

/-- The tolerance of the stored `rstd`: `(D·2^-21 + 2^-140/eps + cond/4)·rstdRef + 2^-140` on a row
with `cond ≤ 1/2` (the roundings, and the mean's error); otherwise `2/sqrt(eps)` (both `rstd` and
`rstdRef` stay below about `1/sqrt(eps)`). -/
noncomputable def rstdTol (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  if cond d init t ≤ 1 / 2 then
    ((depth d : ℝ) * 2 ^ (-21 : ℤ) + 2 ^ (-140 : ℤ) / epsR d + cond d init t / 4) *
      rstdRef d init t + 2 ^ (-140 : ℤ)
  else 2 / Real.sqrt (epsR d)

/-- The bound on the kernel's standardised value `x_hat` against `zRef` (not an output; `yTol` and
`Pre`'s `YRoom` use it): relative on a row with `cond ≤ 1/2`; otherwise from
`rstd ≤ (1 + gam)/sqrt(eps)`. -/
noncomputable def zTol (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  if cond d init t ≤ 1 / 2 then
    (1 + 2 * gam d) * (|zRef d init t j| * (gam d + mTerm d init t) + cond d init t) +
      2 ^ (-140 : ℤ) * (1 + rstdRef d init t)
  else
    |zRef d init t j| +
      (1 + 2 * gam d) * (|xR d init t j - meanRef d init t| + meanTol d init t) /
        Real.sqrt (epsR d) + 2 ^ (-140 : ℤ) * (1 + 1 / Real.sqrt (epsR d))

/-- The magnitude of output `(t, j)`'s terms, `|w j|·|zRef j| + |b j|` (at least `|yRef j|`). -/
noncomputable def yMag (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  |(wv d init j).toReal| * |zRef d init t j| + |(bv d init j).toReal|

/-- The tolerance of each output element: `2^-9·yMag` (the `float16` rounding and the `float32`
steps), the standardised value's error `zTol` scaled by `2·|w|`, and `2^-24`. -/
noncomputable def yTol (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  2 ^ (-9 : ℤ) * yMag d init t j + 2 * |(wv d init j).toReal| * zTol d init t j + 2 ^ (-24 : ℤ)

/-- Row `t`'s output has room in `float16`: for every column, `|w|·(|zRef| + zTol) + |b| ≤ 2^15`
(the bound on the kernel's `float32` `x_hat·w + b` stays below `float16`'s largest value `65504`). -/
def YRoom (d : Inst) (init : Mem) (t : Nat) : Prop :=
  ∀ j, j < d.N → |(wv d init j).toReal| * (|zRef d init t j| + zTol d init t j) +
    |(bv d init j).toReal| ≤ 2 ^ 15

/-- The inputs covered: the six buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), every element of `x`, `weight` and `bias` the launch reads is finite, and every row's
output has room in `float16` (README, Exclusions). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.w.Valid init ∧ d.b.Valid init ∧ d.y.Valid init ∧ d.mean.Valid init ∧
    d.rstd.Valid init ∧
    (∀ t, t < d.M → ∀ j, j < d.N → (xv d init t j).isFinite = true) ∧
    (∀ j, j < d.N → (wv d init j).isFinite = true ∧ (bv d init j).isFinite = true) ∧
    (∀ t, t < d.M → YRoom d init t)

/-- What the launch leaves in row `t < M`'s outputs:
* `mean[t]` is a finite `float32` within `meanTol` of `meanRef`;
* `rstd[t]` is a finite `float32` within `rstdTol` of `rstdRef`;
* `y[t, j]` is a finite `float16` within `yTol j` of `yRef j`. -/
def RowDone (d : Inst) (init fin : Mem) (t : Nat) : Prop :=
  (∃ c : F32, fin.read .f32 (d.mean.ptr (t : Int)).addr = some c ∧ c.isFinite = true ∧
    |c.toReal - meanRef d init t| ≤ meanTol d init t) ∧
  (∃ c : F32, fin.read .f32 (d.rstd.ptr (t : Int)).addr = some c ∧ c.isFinite = true ∧
    |c.toReal - rstdRef d init t| ≤ rstdTol d init t) ∧
  (∀ j, j < d.N → ∃ c : F16, fin.read .f16 (d.y.ptr (idx d t j : Int)).addr = some c ∧
    c.isFinite = true ∧ |c.toReal - yRef d init t j| ≤ yTol d init t j)

/-- What the launch leaves: every row as `RowDone` says, and every byte outside the three outputs'
elements unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.M → RowDone d init fin t) ∧ ∀ z, z ∉ outBytes d → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.MathTotal`: `tl.sqrt` on `float32` has a result for every input (line 81). -/
  sqrt_total : Rules.Num.MathTotal P .sqrt .binary32
  /-- [rule] `Num.SqrtF32`: `tl.sqrt` on `float32` (`sqrt.approx.ftz.f32`), on `0` and inputs of at
  least `2^-126` (line 81's input is at least about the `float32` `eps`). -/
  sqrt_acc : Rules.Num.SqrtF32 P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs (lines 72, 80,
  81). -/
  div_total : Rules.Num.DivTotal P .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32` (lines 72, 80, 81). -/
  div_full : Rules.Num.DivFull P
  /-- [rule] `Num.SumDepth`: the trees of the float `tl.sum`s (lines 72, 80) over their `BLOCK_SIZE`
  positions have depth at most `⌈log₂ BLOCK_SIZE⌉` (`depth`). Neither reduction is in a loop: each
  reduces a loop's carried block after the loop. -/
  sum_depth : Rules.Num.SumDepth P

/-- **The claim.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.ref.mathTotal,
    Platform.sm90ref.sqrtF32, Platform.ref.divTotal _, Platform.ref.divFull,
    Platform.sm90ref.sumDepth⟩⟩

/-- The double `1e-5` (`0x3EE4F8B588E368F1`), the `eps` of the test and the benchmark. -/
def eps1e5 : F64 := FloatV.ofPat 0x3EE4F8B588E368F1

theorem eps1e5_facts : eps1e5.isFinite = true ∧ (2 : ℚ) ^ (-125 : ℤ) ≤ eps1e5.toRat ∧
    eps1e5.toRat ≤ (2 : ℚ) ^ (126 : ℤ) := by
  refine ⟨by decide +kernel, ?_, ?_⟩
  · have : (2 ^ (-125 : ℤ) : ℚ) ≤ (FloatV.ofPat 0x3EE4F8B588E368F1 : F64).toRat := by
      decide +kernel
    exact this
  · have : (FloatV.ofPat 0x3EE4F8B588E368F1 : F64).toRat ≤ (2 ^ (126 : ℤ) : ℚ) := by
      decide +kernel
    exact this

/-! ## The host's layout

`mkInst M N BLOCK` is the host's launch for those sizes (`SETUP.md`, "The launches"): `x`
contiguous `[M, N]`, `weight` and `bias` `[N]`, the fresh `y`, `mean`, `rstd`, `eps = 1e-5`, the six
tensors at `2^36`, `2·2^36`, …, `6·2^36` (each well below `2^36` bytes). -/

def mkInst (M N BLOCK : Nat) : Inst where
  M := M
  N := N
  BLOCK := BLOCK
  eps := eps1e5
  x := ⟨2 ^ 36, M * N⟩
  w := ⟨2 * 2 ^ 36, N⟩
  b := ⟨3 * 2 ^ 36, N⟩
  y := ⟨4 * 2 ^ 36, M * N⟩
  mean := ⟨5 * 2 ^ 36, M⟩
  rstd := ⟨6 * 2 ^ 36, M⟩

/-- Two buffers whose byte ranges do not overlap. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (hbc : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- Elements of two buffers whose byte ranges do not overlap, each inside its buffer. -/
theorem elems_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2) {S S' : Set Nat}
    (hbc : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base)
    (hS : ∀ j ∈ S, j < b.len) (hS' : ∀ j ∈ S', j < c.len) :
    Disjoint (b.elems S) (c.elems S') :=
  Buf.elems_disjoint_of b c (bytes_disj b c hbc) hS hS'

/-- **Every host launch at these sizes is covered**: `2 ≤ N ≤ 32768`, `N` a multiple of `128`,
`BLOCK_SIZE` its next power of two, `1 ≤ M ≤ 2^22`, `M·N ≤ 2^30`, `eps = 1e-5`. -/
theorem mkInst_ok {M N BLOCK : Nat} (hn : 2 ≤ N) (hn' : N ≤ 32768) (hbd : ∃ e, BLOCK = 2 ^ e)
    (hnb : N ≤ BLOCK) (hb2 : BLOCK < 2 * N) (hm : 1 ≤ M) (hmn : M * N ≤ 2 ^ 30)
    (hdim : N % 128 = 0) (hmc : M ≤ 2 ^ 22) :
    (mkInst M N BLOCK).ok := by
  have hM : M ≤ 2 ^ 30 := le_trans (Nat.le_mul_of_pos_right M (by omega)) hmn
  have hMNZ : ((M * N : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hmn
  have hMZ : ((M : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hM
  have hNZ : ((N : Nat) : ℤ) ≤ 32768 := by exact_mod_cast hn'
  have hrows : (M - 1) * N < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.sub_le M 1)) (by omega)
  have mat : ∀ j ∈ matSet (mkInst M N BLOCK), j < M * N := fun j hj => hj
  have row : ∀ j ∈ rowSet (mkInst M N BLOCK), j < M := fun j hj => hj
  have vec : ∀ j ∈ vecSet (mkInst M N BLOCK), j < N := fun j hj => hj
  refine ⟨hn, hn', hbd, hnb, hb2, hm, hrows, ⟨hdim, by simp only [mkInst]; omega⟩, hmc, eps1e5_facts.1, eps1e5_facts.2.1,
    eps1e5_facts.2.2, le_rfl, le_rfl, le_rfl, le_rfl, le_rfl, le_rfl, ?_, ?_, ?_, ?_⟩
  · simp only [outBytes, inBytes]
    refine Disjoint.union_left (Disjoint.union_left ?_ ?_) ?_ <;>
      refine Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_ <;>
      first
        | exact elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) mat mat
        | exact elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) mat vec
        | exact elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) row mat
        | exact elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) row vec
  all_goals first
    | exact elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) mat row
    | exact elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) row row

/-! ## Coverage: the test (source line 374) and the benchmark (lines 326–375) -/

/-- The test's launch: `test_layer_norm(1151, 8192, torch.float16)`, `BLOCK_SIZE = 8192`. -/
def setup_test : Inst := mkInst 1151 8192 8192
theorem setup_test_ok : setup_test.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨13, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- `next_power_of_2(512·i)` for `2 ≤ i ≤ 31`. -/
def benchBlock (i : Nat) : Nat :=
  if i ≤ 2 then 1024 else if i ≤ 4 then 2048 else if i ≤ 8 then 4096 else if i ≤ 16 then 8192
  else 16384

/-- The benchmark's forward launch for `N = 512·i`: `M = 4096`, `float16`. -/
def setup_bench (i : Nat) : Inst := mkInst 4096 (512 * i) (benchBlock i)

/-- **Every forward launch of the benchmark**: `M = 4096`, `N = 512·i` for `2 ≤ i ≤ 31`
(`x_vals`, line 329), `float16`, `eps = 1e-5` (the `triton` provider's `y_fwd()`, line 367). -/
theorem setup_bench_ok (i : Nat) (hi : 2 ≤ i ∧ i ≤ 31) : (setup_bench i).ok := by
  obtain ⟨h1, h2⟩ := hi
  have hpow : ∃ e, benchBlock i = 2 ^ e := by
    unfold benchBlock
    split_ifs
    exacts [⟨10, rfl⟩, ⟨11, rfl⟩, ⟨12, rfl⟩, ⟨13, rfl⟩, ⟨14, rfl⟩]
  refine mkInst_ok (by omega) (by omega) hpow ?_ ?_ (by norm_num) (by omega) (by omega) (by norm_num)
  · unfold benchBlock; split_ifs <;> omega
  · unfold benchBlock; split_ifs <;> omega

/-! ## Satisfiable -/

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

/-- Every aligned `float16` read of `zeroMem` gives `+0`. -/
theorem zeroMem_rval {p : Buf .f16} {i : Nat} (h1 : 0 ≤ (p.ptr (i : Int)).addr)
    (h2 : (p.ptr (i : Int)).addr + 2 ≤ 2 ^ 64) (h3 : (p.ptr (i : Int)).addr % 2 = 0) :
    rval zeroMem p i = DType.ofBits .f16 (TVA.Math.Bytes.leValue [0, 0]) := by
  have hb : zeroMem.readBytes (p.ptr (i : Int)).addr 2 = some [0, 0] := by
    simp [Mem.readBytes, zeroMem]
  have ha : Mem.Aligned .f16 (p.ptr (i : Int)).addr := ⟨h1, h2, h3⟩
  have hrd : zeroMem.read .f16 (p.ptr (i : Int)).addr =
      some (DType.ofBits .f16 (TVA.Math.Bytes.leValue [0, 0])) := by
    unfold Mem.read
    rw [show DType.f16.bytes = 2 from rfl]
    simp only [ha, ite_true, hb, Option.map_some]
  unfold rval
  rw [hrd]

theorem zero16_facts : (DType.ofBits .f16 (TVA.Math.Bytes.leValue [0, 0]) : F16).isFinite = true ∧
    (DType.ofBits .f16 (TVA.Math.Bytes.leValue [0, 0]) : F16).toReal = 0 := by
  refine ⟨by decide +kernel, ?_⟩
  have h : (DType.ofBits .f16 (TVA.Math.Bytes.leValue [0, 0]) : F16).toRat = 0 := by
    decide +kernel
  simp [FloatV.toReal, h]

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hok := setup_test_ok
  refine ⟨setup_test, zeroMem, hok, ?_⟩
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid zeroMem :=
    fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [zeroMem]⟩
  have hidx : ∀ t j, t < 1151 → j < 8192 → t * 8192 + j < 1151 * 8192 := fun t j ht hj => by omega
  have hw : ∀ j, j < 8192 → (wv setup_test zeroMem j).toReal = 0 ∧
      (bv setup_test zeroMem j).toReal = 0 := fun j hj => by
    refine ⟨?_, ?_⟩ <;> simp only [wv, bv] <;> rw [zeroMem_rval] <;> try exact zero16_facts.2
    all_goals simp only [Buf.ptr, setup_test, mkInst, DType.bytes]; push_cast; omega
  refine ⟨hv _ (by simp [setup_test, mkInst]) (by simp [setup_test, mkInst, DType.bytes])
      (by simp [setup_test, mkInst, DType.bytes]),
    hv _ (by simp [setup_test, mkInst]) (by simp [setup_test, mkInst, DType.bytes])
      (by simp [setup_test, mkInst, DType.bytes]),
    hv _ (by simp [setup_test, mkInst]) (by simp [setup_test, mkInst, DType.bytes])
      (by simp [setup_test, mkInst, DType.bytes]),
    hv _ (by simp [setup_test, mkInst]) (by simp [setup_test, mkInst, DType.bytes])
      (by simp [setup_test, mkInst, DType.bytes]),
    hv _ (by simp [setup_test, mkInst]) (by simp [setup_test, mkInst, DType.bytes])
      (by simp [setup_test, mkInst, DType.bytes]),
    hv _ (by simp [setup_test, mkInst]) (by simp [setup_test, mkInst, DType.bytes])
      (by simp [setup_test, mkInst, DType.bytes]),
    fun t ht j hj => ?_, fun j hj => ?_, fun t ht j hj => ?_⟩
  · change t < 1151 at ht
    change j < 8192 at hj
    have h1 := hidx t j ht hj
    unfold xv
    rw [zeroMem_rval]
    · exact zero16_facts.1
    all_goals simp only [Buf.ptr, setup_test, mkInst, idx, DType.bytes]; push_cast; omega
  · change j < 8192 at hj
    refine ⟨?_, ?_⟩ <;> simp only [wv, bv] <;> rw [zeroMem_rval] <;> try exact zero16_facts.1
    all_goals simp only [Buf.ptr, setup_test, mkInst, DType.bytes]; push_cast; omega
  · change j < 8192 at hj
    obtain ⟨h1, h2⟩ := hw j hj
    rw [h1, h2]
    norm_num

end Claims.LayerNormFwdF16
