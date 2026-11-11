import TVA

/-!
# `layer_norm_fwd_bf16`: LayerNorm and GroupNorm forward, fla

The kernel is `layer_norm_fwd_kernel` of `source/layernorm.py` (lines 50–114; it reads the
module-level names `triton` and `tl`, bound by the imports of lines 46–47). Its host
`layer_norm_fwd` (lines 117–182) launches it with one program per `BT = 32` consecutive rows, the
grid `(cdiv(T, 32),)`, `num_warps = 4` (performance-only). The claim covers the paths `SETUP.md`
asks for: fla's `GroupNorm` without a residual (RWKV-6's and RWKV-7's `g_norm`: `G` groups, row `t`
normalised with the weight and bias of group `t % G`), and fla's `LayerNorm` (`G = 1`) with and
without a residual. The flags: `IS_RMS_NORM = False`, `HAS_RESIDUAL = STORE_RESIDUAL_OUT` (either
value), `HAS_WEIGHT = True`, `HAS_BIAS = True`; `bfloat16` tensors with `float32` `mean` and
`rstd`.

What is claimed: every covered launch ends normally, and for every row: with a residual, the
residual out is the `bfloat16` rounding of the `float32` sum `s = x + res`, bit for bit (without
one, `s = x`); the stored mean, `rstd` and every output element are within tolerances of the exact
real LayerNorm of `s` with the weight and bias of the row's group; every other byte is unchanged.
The tolerances hold on every row: they grow with the row's condition number `cond` (the mean's
possible error as the subtractions see it, in units of the row's standard deviation), which `eps`
bounds, `rstdRef ≤ eps^(-1/2)`.

The conditions: `D` from 1 to 512 with `BD` its next power of two; `T` from 1 to `2^31 - 1`; `G`
from 1 to `2^16`; the Python `float` `eps` finite, from `2^-125` to `2^126`; every tensor holds what
the launch touches, and the outputs share no byte with the inputs or with each other; the inputs
are finite, `|x|, |residual| ≤ 2^56` and `|weight|, |bias| ≤ 2^16`.

The tolerances rest on the device facts for `tl.sqrt` and `/` on `float32` and on the depth of a
`tl.sum`'s tree (`Device`).
-/

namespace Claims.LayerNormFwd

open TVA

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `layer_norm_fwd_kernel` (source lines 50–114). The pointers are `*bf16` (`x`, `y`, `w`, `b`,
`res`, `res_out`) and `*fp32` (`mean`, `rstd`), the dtypes of the covered launches; without a
residual the host passes `None` for `res` and `res_out` (a compile-time `None`, never read: the
branches that use them are not taken), which the model's `res_ptr` and `res_out_ptr` stand for.
`eps` is the runtime `float32` scalar (a Python `float` argument: `FloatV.hostF32` of the double,
`model`); `T` is a runtime `int32` (it is in `do_not_specialize`, so never the compile-time
constant `1`); `G`, `D`, `BD` and the flags `HAS_RESIDUAL`, `STORE_RESIDUAL_OUT` are `tl.constexpr`
parameters, Lean parameters here, and their branches are compile-time `if`s, written as Lean `if`s.
The other `tl.constexpr` parameters have one value on every covered launch, which the model
fixes: `BT = 32` (the host's `BT=32`, line 178), `NB` (never read by the body), and
`IS_RMS_NORM = False`, `HAS_WEIGHT = True`, `HAS_BIAS = True` (`SETUP.md`); the branches those
three flags do not take are not modelled (no covered launch takes them). -/
def layer_norm_fwd_kernel (x_ptr y_ptr w_ptr b_ptr res_ptr res_out_ptr : Ptr .bf16)
    (mean_ptr rstd_ptr : Ptr .f32) (eps : F32) (T : I32) (G D BD : Nat)
    (HAS_RESIDUAL STORE_RESIDUAL_OUT : Bool) : Kernel where
  body ctx := do
    -- line 73: the program's row block, converted to `int64` (`tl.convert`)
    let i_t := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 75: the constexpr `BT` (= 32) built in `int64`; the `int32` `tl.arange` converted
    let o_t : Tensor [32] I64 :=
      i_t * (IntV.ofNat 32 : I64) + (← tl.convert .i32 .i64 (tl.arange 0 32))
    -- line 76: `%` of `int64` tensors; the constexpr `G` built in `int64`
    let o_g ← tl.irem o_t (bc (IntV.ofNat G : I64))
    -- line 77
    let o_d : Tensor [BD] I32 := tl.arange 0 BD
    -- line 78: the constexpr `D` built in `int32`, the signed comparison
    let m_d := o_d.ltS (IntV.ofNat D : I32)
    -- line 79: the `int32` scalar `T` converted to `int64`, the signed comparison
    let m_t := o_t.ltS (← tl.convert .i32 .i64 (Tensor.scalar T)).item
    -- line 80: `&` of the `int1` tensors, `m_t[:, None]` and `m_d[None, :]` broadcast
    let m_x := (m_t.col BD).and (m_d.row 32)
    -- line 82: `o_t[:, None] * D` in `int64`; the pointer advanced by it, then by the `int32`
    -- `o_d[None, :]` (left to right; `p + i` advances by the signed value of `i`)
    let p_x := x_ptr + o_t.col BD * (IntV.ofNat D : I64) + o_d.row 32
    -- line 83: `other=0.0` is the `float32` `+0`, converted to the pointer's dtype `bfloat16`;
    -- the loaded `bfloat16` block widens to `float32` exactly (`tl.cvt`)
    let b_x0 : Tensor [32, BD] F32 ← tl.cvt (← tl.load p_x (some m_x)
      (some (← tl.convert .f32 .bf16 (tl.full [32, BD] (FloatV.litF32 (FloatV.ofPat 0))))))
    -- line 84: `if HAS_RESIDUAL:`, a compile-time `if`
    let b_x : Tensor [32, BD] F32 ← if HAS_RESIDUAL then do
        -- line 85
        let p_res := res_ptr + o_t.col BD * (IntV.ofNat D : I64) +
          o_d.row 32
        -- line 86: `b_x += …`: the IEEE `+` (no product feeds it)
        tl.add b_x0 (← tl.cvt (← tl.load p_res (some m_x)
          (some (← tl.convert .f32 .bf16 (tl.full [32, BD] (FloatV.litF32 (FloatV.ofPat 0)))))))
      else pure b_x0
    -- line 87: `if STORE_RESIDUAL_OUT:`, a compile-time `if`
    if STORE_RESIDUAL_OUT then do
      -- line 88
      let p_res_out := res_out_ptr + o_t.col BD * (IntV.ofNat D : I64) +
        o_d.row 32
      -- line 89: `b_x.to(bfloat16)`, one rounding to nearest even (no `fp_downcast_rounding`)
      let b_x16 : Tensor [32, BD] BF16 ← tl.cvt b_x
      tl.store p_res_out b_x16 (some m_x)
    -- line 90: `not IS_RMS_NORM` is `True` on every covered launch, the branch is visited
    -- line 91: `tl.sum(axis=1)` of the `float32` block (no product feeds it), then `/` by the
    -- constexpr `D`, the Python number built in the division's dtype `float32`
    let b_mean : Tensor [32] F32 ←
      tl.div (← tl.sum1 b_x) (bc (FloatV.litF32 (FloatV.ofRat (D : ℚ))))
    -- line 92
    let p_mean := mean_ptr + o_t
    -- line 93: `b_mean` is already `float32`, the pointer's dtype: the `.to` adds no step
    tl.store p_mean b_mean (some m_t)
    -- line 94: `b_mean` is a `float32` quotient (line 91), a product for the fusion rule, so the
    -- `-` it feeds is the fusable `tl.subc`; the lanes `d ≥ D` hold the constant `0.0`, built in
    -- `float32`
    let b_xbar := tl.where (m_d.row 32) (← tl.subc b_x (b_mean.col BD))
      (bc (FloatV.litF32 (FloatV.ofPat 0)))
    -- line 95: the squares' product feeds the sum (`tl.sum1c`), then `/` by `D` as line 91
    let b_var : Tensor [32] F32 ←
      tl.div (← tl.sum1c (← tl.mul b_xbar b_xbar)) (bc (FloatV.litF32 (FloatV.ofRat (D : ℚ))))
    -- lines 96–98: the `else` branch (`IS_RMS_NORM`): not modelled, never taken
    -- line 99: the constant `1` built in `float32`; `b_var + eps` the fusable `tl.addc` (at
    -- `D = 1`, `/ D` is `/ 1.0`, which MLIR folds away, so `b_var` is the product
    -- `b_xbar * b_xbar` itself and the addition may be contracted into an `fma`); `tl.sqrt` the
    -- platform's square root; `/` the platform's division
    let b_rstd : Tensor [32] F32 ←
      tl.div (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
        (← tl.sqrt (← tl.addc b_var (bc eps)))
    -- line 101
    let p_rstd := rstd_ptr + o_t
    -- line 102
    tl.store p_rstd b_rstd (some m_t)
    -- line 104: `HAS_WEIGHT` is `True`, the branch is visited
    -- line 105: the mask is `m_d` only and there is no `other`: the lanes `d ≥ D` hold
    -- unspecified values, which no stored lane uses; every row's group `o_g` is below `G`
    let p_w := w_ptr + o_g.col BD * (IntV.ofNat D : I64) + o_d.row 32
    let b_w : Tensor [32, BD] F32 ← tl.cvt (← tl.load p_w (some (m_d.row 32)))
    -- line 106: `HAS_BIAS` is `True`, the branch is visited
    -- line 107
    let p_b := b_ptr + o_g.col BD * (IntV.ofNat D : I64) + o_d.row 32
    let b_b : Tensor [32, BD] F32 ← tl.cvt (← tl.load p_b (some (m_d.row 32)))
    -- line 108: `not IS_RMS_NORM` is `True`: `(b_x - b_mean[:, None]) * b_rstd[:, None]`, the
    -- fusable `tl.subc` (the quotient `b_mean` feeds it, as at line 94; a choice of its own), then
    -- the IEEE `*`
    let b_x_hat : Tensor [32, BD] F32 ← tl.mul (← tl.subc b_x (b_mean.col BD)) (b_rstd.col BD)
    -- line 109: `HAS_WEIGHT` is `True`: the product `b_x_hat * b_w`
    let b_y0 : Tensor [32, BD] F32 ← tl.mul b_x_hat b_w
    -- line 110: `HAS_BIAS` is `True`, the branch is visited
    -- line 111: `b_y + b_b` where a product feeds `b_y` (line 109): the fusable `tl.addc`
    let b_y : Tensor [32, BD] F32 ← tl.addc b_y0 b_b
    -- line 113
    let p_y := y_ptr + o_t.col BD * (IntV.ofNat D : I64) + o_d.row 32
    -- line 114: `b_y.to(bfloat16)`, one rounding to nearest even
    let b_y16 : Tensor [32, BD] BF16 ← tl.cvt b_y
    tl.store p_y b_y16 (some m_x)

/-! ## The conditions -/

/-- What the host chooses: one launch of `layer_norm_fwd_kernel` through `layer_norm_fwd`
(source lines 117–182) with the covered flags. The tensors are contiguous (`x`, `residual`, `y`,
`res_out` are `[T, D]`, `weight` and `bias` `[G·D]`, `mean` and `rstd` `[T]`), given by their
storage. -/
structure Inst where
  /-- `T` (`x.shape[0]`): the rows (fla's `LayerNormFunction` reshapes its input to
  `[-1, hidden // num_groups]`, so with groups a row is one group of one token). -/
  T : Nat
  /-- `D` (`x.shape[1]`): the row length. -/
  D : Nat
  /-- `BD`: the host passes `min(65536 // 2, triton.next_power_of_2(D))` (line 147). -/
  BD : Nat
  /-- `G` (`num_groups`): row `t` uses the weights and biases of group `t % G` (line 76). -/
  G : Nat
  /-- `residual is not None`: the host's `HAS_RESIDUAL`, and its `STORE_RESIDUAL_OUT`
  (`res_out` is allocated exactly when there is a residual, line 139: `residual_dtype` is the
  residual's dtype then, and `None` otherwise). -/
  hasRes : Bool
  /-- `eps`, the Python `float` (a double; the module's default is `1e-5`). -/
  eps : F64
  /-- `x`, `bfloat16`. -/
  x : Buf .bf16
  /-- `residual`, `bfloat16` (`hasRes`; otherwise the host passes `None` and the buffer is not
  touched). -/
  res : Buf .bf16
  /-- `weight`, `bfloat16`, `G·D` elements. -/
  w : Buf .bf16
  /-- `bias`, `bfloat16`, `G·D` elements. -/
  b : Buf .bf16
  /-- `y` (`torch.empty_like(x)`, line 138), `bfloat16`. -/
  y : Buf .bf16
  /-- `res_out` (`torch.empty(T, D, …)`, line 140), `bfloat16` (`hasRes`; otherwise `None`). -/
  res_out : Buf .bf16
  /-- `mean` (`torch.empty((T,), dtype=torch.float)`, line 143), `float32`. -/
  mean : Buf .f32
  /-- `rstd` (line 144), `float32`. -/
  rstd : Buf .f32

/-- Element `(t, j)` of a contiguous `[T, D]` tensor. -/
def idx (d : Inst) (t j : Nat) : Nat := t * d.D + j

/-- Element `j` of row `t`'s group in `weight` and `bias`: `(t % G)·D + j` (lines 76, 105, 107). -/
def gidx (d : Inst) (t j : Nat) : Nat := (t % d.G) * d.D + j

/-- The elements of a `[T, D]` tensor. -/
def matSet (d : Inst) : Set Nat := {k | k < d.T * d.D}

/-- The elements of `residual` and `res_out` the launch touches: those of a `[T, D]` tensor with a
residual, none without. -/
def resSet (d : Inst) : Set Nat := {k | d.hasRes = true ∧ k < d.T * d.D}

/-- The elements of a `[T]` tensor. -/
def rowSet (d : Inst) : Set Nat := {k | k < d.T}

/-- The elements of a `[G·D]` tensor. -/
def vecSet (d : Inst) : Set Nat := {k | k < d.G * d.D}

/-- The bytes the launch reads. -/
def inBytes (d : Inst) : Set ℤ :=
  d.x.elems (matSet d) ∪ d.res.elems (resSet d) ∪ d.w.elems (vecSet d) ∪ d.b.elems (vecSet d)

/-- The bytes the launch writes. -/
def outBytes (d : Inst) : Set ℤ :=
  d.y.elems (matSet d) ∪ d.res_out.elems (resSet d) ∪ d.mean.elems (rowSet d) ∪
    d.rstd.elems (rowSet d)

/-- The instances covered: `D` from 1 to 512 (this kernel's path), `BD` its next power of two,
`T` from 1 to `2^31 - 1`, `G` from 1 to `2^16`, with or without a residual, `eps` a double in
`[2^-125, 2^126]`, every tensor holding what the launch touches, and the outputs sharing no byte
with the inputs or with each other. -/
structure Inst.ok (d : Inst) : Prop where
  /-- A row has at least one column. -/
  d_pos : 1 ≤ d.D
  /-- `D ≤ 512`: this kernel's path (the host's assert, line 151; fla sends wider rows to
  `layer_norm_fwd_kernel_row`). -/
  d_le : d.D ≤ 512
  /-- `BD` is a power of two. -/
  bd_pow2 : ∃ e, d.BD = 2 ^ e
  /-- `D ≤ BD`. -/
  d_le_bd : d.D ≤ d.BD
  /-- `BD < 2·D`: with `bd_pow2` and `d_le_bd`, `BD = next_power_of_2(D)` (line 147). -/
  bd_lt : d.BD < 2 * d.D
  /-- At least one row. -/
  t_pos : 1 ≤ d.T
  /-- `T < 2^31`: the host passes `T` as a Python `int`, a runtime `int32` below `2^31` (the model's
  `T : I32`). -/
  t_lt : d.T < 2 ^ 31
  /-- At least one group. -/
  g_pos : 1 ≤ d.G
  /-- At most `2^16` groups (README, Exclusions: real use has one group per head). -/
  g_le : d.G ≤ 2 ^ 16
  /-- `eps` is finite. -/
  eps_fin : d.eps.isFinite = true
  /-- `eps ≥ 2^-125` (README, Exclusions: below `2^-126` the kernel's `float32` `eps` is
  subnormal and `tl.sqrt` flushes `var + eps` of a constant row to `0`). -/
  eps_ge : (2 : ℚ) ^ (-125 : ℤ) ≤ d.eps.toRat
  /-- `eps ≤ 2^126` (README, Exclusions). -/
  eps_le : d.eps.toRat ≤ (2 : ℚ) ^ (126 : ℤ)
  /-- `x` holds `T·D` elements. -/
  x_len : d.T * d.D ≤ d.x.len
  /-- With a residual, `residual` holds `T·D` elements. -/
  res_len : d.hasRes = true → d.T * d.D ≤ d.res.len
  /-- `weight` holds `G·D` elements (the host's assert, line 134). -/
  w_len : d.G * d.D ≤ d.w.len
  /-- `bias` holds `G·D` elements (the host's assert, line 136). -/
  b_len : d.G * d.D ≤ d.b.len
  /-- `y` holds `T·D` elements. -/
  y_len : d.T * d.D ≤ d.y.len
  /-- With a residual, `res_out` holds `T·D` elements. -/
  res_out_len : d.hasRes = true → d.T * d.D ≤ d.res_out.len
  /-- `mean` holds `T` elements. -/
  mean_len : d.T ≤ d.mean.len
  /-- `rstd` holds `T` elements. -/
  rstd_len : d.T ≤ d.rstd.len
  /-- The outputs share no byte with the inputs (`torch.empty` allocates them). -/
  out_in : Disjoint (outBytes d) (inBytes d)
  /-- `y` and `res_out` share no byte. -/
  y_res_out : Disjoint (d.y.elems (matSet d)) (d.res_out.elems (resSet d))
  /-- `y` and `mean` share no byte. -/
  y_mean : Disjoint (d.y.elems (matSet d)) (d.mean.elems (rowSet d))
  /-- `y` and `rstd` share no byte. -/
  y_rstd : Disjoint (d.y.elems (matSet d)) (d.rstd.elems (rowSet d))
  /-- `res_out` and `mean` share no byte. -/
  res_out_mean : Disjoint (d.res_out.elems (resSet d)) (d.mean.elems (rowSet d))
  /-- `res_out` and `rstd` share no byte. -/
  res_out_rstd : Disjoint (d.res_out.elems (resSet d)) (d.rstd.elems (rowSet d))
  /-- `mean` and `rstd` share no byte. -/
  mean_rstd : Disjoint (d.mean.elems (rowSet d)) (d.rstd.elems (rowSet d))

/-- The model on an instance: the buffers' first elements, `eps` as the launcher's `float32`
(`FloatV.hostF32` of the Python double), `T` as an `int32`, the constexprs `G`, `D`, `BD`, and
`HAS_RESIDUAL = STORE_RESIDUAL_OUT = hasRes` (lines 174–175). -/
def model (d : Inst) : Kernel :=
  layer_norm_fwd_kernel (d.x.ptr 0) (d.y.ptr 0) (d.w.ptr 0) (d.b.ptr 0) (d.res.ptr 0)
    (d.res_out.ptr 0) (d.mean.ptr 0) (d.rstd.ptr 0) (FloatV.hostF32 d.eps) (IntV.ofNat d.T)
    d.G d.D d.BD d.hasRes d.hasRes

/-- The host's launch (lines 154–180): the grid `(cdiv(T, 32),)`. The source's block shapes are
`[32]` (`o_t`, `o_g`, `m_t`, `b_mean`, `b_var`, `b_rstd`, the `mean` and `rstd` pointers), `[BD]`
(`o_d`, `m_d`), `[32, 1]` (`o_t[:, None]`, `o_g[:, None]`, `b_mean[:, None]`, `b_rstd[:, None]`),
`[1, BD]` (`o_d[None, :]`, `m_d[None, :]`) and `[32, BD]` (the loads, the arithmetic, the stores).
`enable_fp_fusion` is Triton's default (`true`); `grid1` keeps `num_ctas = 1`, `launch_pdl = False`
and `debug = false`; `num_warps = 4` is performance-only. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.T + 31) / 32) [[32], [d.BD], [32, 1], [1, d.BD], [32, d.BD]]

/-- The `bfloat16` element `i` of a buffer in memory `m` (`+0` where absent; `Pre` makes every
element the launch reads present). -/
def rval (m : Mem) (p : Buf .bf16) (i : Nat) : BF16 :=
  match m.read .bf16 (p.ptr (i : Int)).addr with
  | some v => v
  | none => FloatV.ofPat 0

/-- A finite `bfloat16` of magnitude at most `B`. -/
def InOkB (B : ℚ) (a : BF16) : Prop := a.isFinite = true ∧ |a.toRat| ≤ B

/-- An element of `x` or `residual` covered: a finite `bfloat16` of magnitude at most `2^56`
(README, Exclusions: the kernel's `float32` sum of squares overflows from `2^59`). -/
def InOk (a : BF16) : Prop := InOkB (2 ^ 56) a

/-- An element of `weight` or `bias` covered: a finite `bfloat16` of magnitude at most `2^16`
(README, Exclusions). -/
def InOkW (a : BF16) : Prop := InOkB (2 ^ 16) a

/-- `x[t, j]`, `residual[t, j]`, and `weight`, `bias` at column `j` of row `t`'s group, in the
initial memory. -/
def xv (d : Inst) (init : Mem) (t j : Nat) : BF16 := rval init d.x (idx d t j)
def rv (d : Inst) (init : Mem) (t j : Nat) : BF16 := rval init d.res (idx d t j)
def wv (d : Inst) (init : Mem) (t j : Nat) : BF16 := rval init d.w (gidx d t j)
def bv (d : Inst) (init : Mem) (t j : Nat) : BF16 := rval init d.b (gidx d t j)

/-- The inputs covered: the buffers the launch touches are valid (aligned, inside `[0, 2^64)`,
every byte present: `residual` and `res_out` only with a residual), and every element of `x`,
`residual` (with a residual) the launch reads is `InOk`, and every element of `weight` and `bias`
is `InOkW`. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ (d.hasRes = true → d.res.Valid init) ∧ d.w.Valid init ∧ d.b.Valid init ∧
    d.y.Valid init ∧ (d.hasRes = true → d.res_out.Valid init) ∧ d.mean.Valid init ∧
    d.rstd.Valid init ∧
    (∀ t, t < d.T → ∀ j, j < d.D →
      InOk (xv d init t j) ∧ (d.hasRes = true → InOk (rv d init t j))) ∧
    (∀ k, k < d.G * d.D → InOkW (rval init d.w k) ∧ InOkW (rval init d.b k))

/-! ## The specification

For row `t`, from the host's inputs: `s j` is the `float32` sum `x[t, j] + residual[t, j]` of the
two `bfloat16` values (the IEEE addition, as the host's reference forms it) with a residual, and
`x[t, j]` widened to `float32` (exact) without; the reference is the exact real LayerNorm of `s`:
`meanRef = Σ s / D`, `varRef = Σ (s − meanRef)² / D`, `rstdRef = 1 / sqrt(varRef + eps)` (the Python
double `eps`), `zRef j = (s j − meanRef)·rstdRef` and `yRef j = zRef j · w j + b j`, with `w j`,
`b j` the weight and bias of row `t`'s group `t % G` at column `j`. The tolerances (README,
Tolerances, for their derivation), with `L = log₂ BD` (`depth`, the depth of the reductions' trees
over their `BD` positions, `Device.SumDepth`):

* `meanTol = (L + 4)·2^-23 · Σ|s|/D + 2^-140`;
* `eTol j = meanTol + 2^-23·(|s j| + Σ|s|/D)`, the mean a lane's fusable subtraction sees;
* `cond = (meanTol + 2^-23·(Σ|s|/D + rmsS))·rstdRef`, the row's condition number
  (`rmsS = sqrt(Σ s²/D)`);
* `gam = (L + 10)·2^-23 + 2^-136 / eps`, the relative error of `rstd` from rounding;
* `rstdTol = 2·(cond + gam)·rstdRef` when `4·cond ≤ 1`, else `2·eps^(-1/2)` (an ill-conditioned
  row: the variance may be lost to the mean's rounding, and `rstd` is only known to be at most
  `eps^(-1/2)`);
* `zTol j = |s j − meanRef|·rstdTol + eTol j·(rstdRef + rstdTol)`, the standardised value's error;
* `yTol j = 2^-7·|yRef j| + 4·|w j|·zTol j + 2^-21·|b j| + 2^-120`.
-/

/-- `s j`: with a residual, the `float32` sum of row `t`'s two input elements at column `j`;
without, `x[t, j]` widened to `float32`. -/
def sv (d : Inst) (init : Mem) (t j : Nat) : F32 :=
  if d.hasRes then FloatV.add (FloatV.cvt (xv d init t j)) (FloatV.cvt (rv d init t j))
  else FloatV.cvt (xv d init t j)

/-- `s j` as a real. -/
noncomputable def sR (d : Inst) (init : Mem) (t j : Nat) : ℝ := (sv d init t j).toReal

/-- The row's exact mean `Σ_j s j / D`. -/
noncomputable def meanRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.D, sR d init t j) / d.D

/-- The row's exact mean magnitude `Σ_j |s j| / D`. -/
noncomputable def absMean (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.D, |sR d init t j|) / d.D

/-- The row's exact variance `Σ_j (s j − meanRef)² / D`. -/
noncomputable def varRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.D, (sR d init t j - meanRef d init t) ^ 2) / d.D

/-- The host's `eps` (the double) as a real. -/
noncomputable def epsR (d : Inst) : ℝ := d.eps.toReal

/-- The exact `1 / sqrt(varRef + eps)`. -/
noncomputable def rstdRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  1 / Real.sqrt (varRef d init t + epsR d)

/-- The exact standardised value `(s j − meanRef)·rstdRef`. -/
noncomputable def zRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  (sR d init t j - meanRef d init t) * rstdRef d init t

/-- The exact output `zRef j · w j + b j`, with the weight and bias of row `t`'s group. -/
noncomputable def yRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  zRef d init t j * (wv d init t j).toReal + (bv d init t j).toReal

/-- `L = ⌈log₂ BD⌉` (`log₂ BD`, a power of two): the largest depth of the trees of the two
`tl.sum`s over their `BD` positions on sm_90 (`Device.SumDepth`). -/
def depth (d : Inst) : Nat := Nat.clog 2 d.BD

/-- The tolerance of the stored mean. -/
noncomputable def meanTol (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  ((depth d : ℝ) + 4) * 2 ^ (-23 : ℤ) * absMean d init t + 2 ^ (-140 : ℤ)

/-- The row's root mean square `sqrt(Σ_j (s j)² / D)`. -/
noncomputable def rmsS (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  Real.sqrt ((∑ j ∈ Finset.range d.D, sR d init t j ^ 2) / d.D)

/-- The bound on the mean a lane's subtraction (lines 94, 108) sees, against `meanRef`: the stored
mean's tolerance, and the half unit the fusable subtraction may move the mean or `s j` by. -/
noncomputable def eTol (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  meanTol d init t + 2 ^ (-23 : ℤ) * (|sR d init t j| + absMean d init t)

/-- The row's condition number: the lanes' mean errors (`eTol`, their root mean square at most
`meanTol + 2^-23·(Σ|s|/D + rmsS)`) in units of the row's standard deviation (with `eps`). -/
noncomputable def cond (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t)) * rstdRef d init t

/-- The relative error of `rstd` from the roundings of the variance, `eps`, `tl.sqrt` and `/`. -/
noncomputable def gam (d : Inst) : ℝ :=
  ((depth d : ℝ) + 10) * 2 ^ (-23 : ℤ) + 2 ^ (-136 : ℤ) / epsR d

/-- The tolerance of the stored `rstd`: `2·(cond + gam)·rstdRef` on a well-conditioned row
(`4·cond ≤ 1`), else `2·eps^(-1/2)` (the kernel's `rstd` is then only known to lie in
`[0, eps^(-1/2)]`, up to rounding). -/
noncomputable def rstdTol (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  if 4 * cond d init t ≤ 1 then 2 * (cond d init t + gam d) * rstdRef d init t
  else 2 / Real.sqrt (epsR d)

/-- The bound on the kernel's standardised value `b_x_hat` against `zRef` (not an output; it
makes `yTol` readable): `rstd`'s error on the deviation, and the lane's mean error scaled by
`rstd`. -/
noncomputable def zTol (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  |sR d init t j - meanRef d init t| * rstdTol d init t +
    eTol d init t j * (rstdRef d init t + rstdTol d init t)

/-- The tolerance of each output element: the output's `bfloat16` rounding (`2^-7`, twice it),
the standardised value's error through the weight, and the bias's `float32` rounding. -/
noncomputable def yTol (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  2 ^ (-7 : ℤ) * |yRef d init t j| + 4 * |(wv d init t j).toReal| * zTol d init t j +
    2 ^ (-21 : ℤ) * |(bv d init t j).toReal| + 2 ^ (-120 : ℤ)

/-- What the launch leaves in row `t < T`'s outputs:
* with a residual, `res_out[t, j]` is the `bfloat16` rounding (to nearest even) of `s j`, bit for
  bit;
* `mean[t]` is a finite `float32` within `meanTol` of `meanRef`;
* `rstd[t]` is a finite `float32` within `rstdTol` of `rstdRef`;
* `y[t, j]` is a finite `bfloat16` within `yTol j` of `yRef j`. -/
def RowDone (d : Inst) (init fin : Mem) (t : Nat) : Prop :=
  (d.hasRes = true → ∀ j, j < d.D →
    fin.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr = some (FloatV.cvt (sv d init t j))) ∧
  (∃ c : F32, fin.read .f32 (d.mean.ptr (t : Int)).addr = some c ∧ c.isFinite = true ∧
    |c.toReal - meanRef d init t| ≤ meanTol d init t) ∧
  (∃ c : F32, fin.read .f32 (d.rstd.ptr (t : Int)).addr = some c ∧ c.isFinite = true ∧
    |c.toReal - rstdRef d init t| ≤ rstdTol d init t) ∧
  (∀ j, j < d.D → ∃ c : BF16, fin.read .bf16 (d.y.ptr (idx d t j : Int)).addr = some c ∧
    c.isFinite = true ∧ |c.toReal - yRef d init t j| ≤ yTol d init t j)

/-- What the launch leaves: every row as `RowDone` says, and every byte outside the outputs'
elements the launch writes (`outBytes`) unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.T → RowDone d init fin t) ∧ ∀ z, z ∉ outBytes d → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `layer_norm_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The double `1e-5` (`0x3EE4F8B588E368F1`), the module's default `eps`. -/
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

/-- The double `64 * 1e-5` (`0x3F44F8B588E368F1`, Python's `0.00064`): RWKV-7's `GroupNorm`
`eps = head_dim * norm_eps` (`fla/layers/rwkv7.py:131`, `head_dim = 64`, `norm_eps = 1e-5`). -/
def eps64e5 : F64 := FloatV.ofPat 0x3F44F8B588E368F1

theorem eps64e5_facts : eps64e5.isFinite = true ∧ (2 : ℚ) ^ (-125 : ℤ) ≤ eps64e5.toRat ∧
    eps64e5.toRat ≤ (2 : ℚ) ^ (126 : ℤ) := by
  refine ⟨by decide +kernel, ?_, ?_⟩
  · have : (2 ^ (-125 : ℤ) : ℚ) ≤ (FloatV.ofPat 0x3F44F8B588E368F1 : F64).toRat := by
      decide +kernel
    exact this
  · have : (FloatV.ofPat 0x3F44F8B588E368F1 : F64).toRat ≤ (2 ^ (126 : ℤ) : ℚ) := by
      decide +kernel
    exact this

/-- A launch of the host's: `T` rows of `D` columns, `BD = next_power_of_2(D)`, `G` groups, with
(`hr = true`) or without a residual, the double `e` as `eps`; the eight tensors at `2^36`,
`2·2^36`, … (each well below `2^36` bytes). -/
def mkInst (T D BD G : Nat) (hr : Bool) (e : F64) : Inst where
  T := T
  D := D
  BD := BD
  G := G
  hasRes := hr
  eps := e
  x := ⟨2 ^ 36, T * D⟩
  res := ⟨2 * 2 ^ 36, T * D⟩
  w := ⟨3 * 2 ^ 36, G * D⟩
  b := ⟨4 * 2 ^ 36, G * D⟩
  y := ⟨5 * 2 ^ 36, T * D⟩
  res_out := ⟨6 * 2 ^ 36, T * D⟩
  mean := ⟨7 * 2 ^ 36, T⟩
  rstd := ⟨8 * 2 ^ 36, T⟩

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

theorem mkInst_ok {T D BD G : Nat} {hr : Bool} {e : F64} (hd : 1 ≤ D) (hd' : D ≤ 512)
    (hbd : ∃ k, BD = 2 ^ k) (hdbd : D ≤ BD) (hbd2 : BD < 2 * D) (ht : 1 ≤ T)
    (htd : T * D ≤ 2 ^ 30) (hg : 1 ≤ G) (hg' : G ≤ 2 ^ 16)
    (he : e.isFinite = true ∧ (2 : ℚ) ^ (-125 : ℤ) ≤ e.toRat ∧ e.toRat ≤ (2 : ℚ) ^ (126 : ℤ)) :
    (mkInst T D BD G hr e).ok := by
  have hT : T ≤ 2 ^ 30 := le_trans (Nat.le_mul_of_pos_right T hd) htd
  have hgd : G * D ≤ 2 ^ 25 := by
    calc G * D ≤ 2 ^ 16 * 512 := Nat.mul_le_mul hg' hd'
      _ = 2 ^ 25 := by norm_num
  have hTZ : ((T * D : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast htd
  have hTZ' : ((T : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hT
  have hGZ : ((G * D : Nat) : ℤ) ≤ 2 ^ 25 := by exact_mod_cast hgd
  push_cast at hTZ hGZ
  have mat : ∀ j ∈ matSet (mkInst T D BD G hr e), j < T * D := fun j hj => hj
  have res : ∀ j ∈ resSet (mkInst T D BD G hr e), j < T * D := fun j hj => hj.2
  have row : ∀ j ∈ rowSet (mkInst T D BD G hr e), j < T := fun j hj => hj
  have vec : ∀ j ∈ vecSet (mkInst T D BD G hr e), j < G * D := fun j hj => hj
  refine ⟨hd, hd', hbd, hdbd, hbd2, ht, show T < 2 ^ 31 by omega, hg, hg', he.1, he.2.1, he.2.2,
    le_rfl, fun _ => le_rfl, le_rfl, le_rfl, le_rfl, fun _ => le_rfl, le_rfl, le_rfl, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [outBytes, inBytes]
    refine Disjoint.union_left (Disjoint.union_left (Disjoint.union_left ?_ ?_) ?_) ?_ <;>
      refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_ <;>
      refine elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) ?_ ?_ <;>
      first | exact mat | exact res | exact row | exact vec
  all_goals
    refine elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) ?_ ?_ <;>
      first | exact mat | exact res | exact row

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

/-- Every aligned `bfloat16` read of `zeroMem` gives `+0`, a finite value of magnitude `0`. -/
theorem zeroMem_inOk {B : ℚ} (hB : 0 ≤ B) {p : Buf .bf16} {i : Nat} (h1 : 0 ≤ (p.ptr (i : Int)).addr)
    (h2 : (p.ptr (i : Int)).addr + 2 ≤ 2 ^ 64) (h3 : (p.ptr (i : Int)).addr % 2 = 0) :
    InOkB B (rval zeroMem p i) := by
  have hb : zeroMem.readBytes (p.ptr (i : Int)).addr 2 = some [0, 0] := by
    simp [Mem.readBytes, zeroMem]
  have ha : Mem.Aligned .bf16 (p.ptr (i : Int)).addr := ⟨h1, h2, h3⟩
  have hrd : zeroMem.read .bf16 (p.ptr (i : Int)).addr =
      some (DType.ofBits .bf16 (Bytes.leValue [0, 0])) := by
    unfold Mem.read
    rw [show DType.bf16.bytes = 2 from rfl]
    simp only [ha, ite_true, hb, Option.map_some]
  unfold rval
  rw [hrd]
  refine ⟨by decide +kernel, ?_⟩
  have h : (DType.ofBits .bf16 (Bytes.leValue [0, 0]) : BF16).toRat = 0 := by
    decide +kernel
  rw [h]; simpa using hB

theorem mkInst_pre {T D BD G : Nat} {hr : Bool} {e : F64} (htd : T * D ≤ 2 ^ 30) (hT : T ≤ 2 ^ 30)
    (hgd : G * D ≤ 2 ^ 30) : Pre (mkInst T D BD G hr e) zeroMem := by
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid zeroMem :=
    fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [zeroMem]⟩
  have hTZ : ((T * D : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast htd
  have hTZ' : ((T : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hT
  have hGZ : ((G * D : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hgd
  have hidx : ∀ t j, t < T → j < D → t * D + j < T * D := fun t j ht hj =>
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left hj _)
      (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ ht)
  refine ⟨hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    fun _ => hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    fun _ => hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    fun t ht j hj => ?_, fun k hk => ?_⟩
  · change t < T at ht
    change j < D at hj
    have h1 : ((t * D + j : Nat) : ℤ) < ((T * D : Nat) : ℤ) := by exact_mod_cast hidx t j ht hj
    push_cast at h1
    have h0 : (0 : ℤ) ≤ (t : ℤ) * (D : ℤ) := by positivity
    refine ⟨?_, fun _ => ?_⟩ <;> apply zeroMem_inOk (by norm_num) <;>
      simp only [Buf.ptr, mkInst, idx, DType.bytes] <;> push_cast <;> omega
  · change k < G * D at hk
    have h1 : ((k : Nat) : ℤ) < ((G * D : Nat) : ℤ) := by exact_mod_cast hk
    push_cast at h1 hGZ
    constructor <;> apply zeroMem_inOk (by norm_num) <;>
      simp only [Buf.ptr, mkInst, DType.bytes] <;> push_cast <;> omega

theorem claim_satisfiable : claim.Satisfiable :=
  ⟨mkInst 2048 64 64 1 true eps1e5, zeroMem,
    mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) eps1e5_facts,
    mkInst_pre (by norm_num) (by norm_num) (by norm_num)⟩

/-! ## Coverage

The host's test launches (source lines 203–213): `LayerNorm` with a residual, `G = 1`,
`eps = 1e-5`. -/

def setup_1x1 : Inst := mkInst 1 1 1 1 true eps1e5
def setup_1x64 : Inst := mkInst 1 64 64 1 true eps1e5
def setup_31x50 : Inst := mkInst 31 50 64 1 true eps1e5
def setup_2048x50 : Inst := mkInst 2048 50 64 1 true eps1e5
def setup_2048x64 : Inst := mkInst 2048 64 64 1 true eps1e5
def setup_2048x128 : Inst := mkInst 2048 128 128 1 true eps1e5
def setup_33x512 : Inst := mkInst 33 512 512 1 true eps1e5
def setup_1000x300 : Inst := mkInst 1000 300 512 1 true eps1e5
def setup_65536x256 : Inst := mkInst 65536 256 256 1 true eps1e5

theorem setup_1x1_ok : setup_1x1.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨0, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_1x64_ok : setup_1x64.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_31x50_ok : setup_31x50.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_2048x50_ok : setup_2048x50.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_2048x64_ok : setup_2048x64.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_2048x128_ok : setup_2048x128.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_33x512_ok : setup_33x512.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_1000x300_ok : setup_1000x300.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_65536x256_ok : setup_65536x256.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨8, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts

/-! `GroupNorm` without a residual (`SETUP.md`): RWKV-7's `g_norm` (32 heads of 64 at hidden
size 2048, `eps = 64·1e-5`) on a training batch of 4 × 4096 tokens and on one decoding token;
RWKV-6's `g_norm` (4 heads of 512, `eps = 1e-5`) on 4 × 4096 tokens; the shapes of fla's
`GroupNorm` test (`tests/modules/test_layernorm.py:57-62`, 2 × 512 tokens, every hidden size on
this path: `G = 1` with hidden 64, 128, 512 and `G = 4` with hidden 64, 128, 512, 1024, 2048), in
`bfloat16` (the test runs `float32`). -/

def setup_rwkv7_gn_train : Inst := mkInst 524288 64 64 32 false eps64e5
def setup_rwkv7_gn_decode : Inst := mkInst 32 64 64 32 false eps64e5
def setup_rwkv6_gn_train : Inst := mkInst 65536 512 512 4 false eps1e5
def setup_gn_test_512_g4 : Inst := mkInst 4096 128 128 4 false eps1e5
def setup_gn_test_2048_g4 : Inst := mkInst 4096 512 512 4 false eps1e5
def setup_gn_test_64_g1 : Inst := mkInst 1024 64 64 1 false eps1e5
def setup_gn_test_128_g1 : Inst := mkInst 1024 128 128 1 false eps1e5
def setup_gn_test_512_g1 : Inst := mkInst 1024 512 512 1 false eps1e5
def setup_gn_test_64_g4 : Inst := mkInst 4096 16 16 4 false eps1e5
def setup_gn_test_128_g4 : Inst := mkInst 4096 32 32 4 false eps1e5
def setup_gn_test_1024_g4 : Inst := mkInst 4096 256 256 4 false eps1e5

theorem setup_rwkv7_gn_train_ok : setup_rwkv7_gn_train.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps64e5_facts
theorem setup_rwkv7_gn_decode_ok : setup_rwkv7_gn_decode.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps64e5_facts
theorem setup_rwkv6_gn_train_ok : setup_rwkv6_gn_train.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_512_g4_ok : setup_gn_test_512_g4.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_2048_g4_ok : setup_gn_test_2048_g4.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_64_g1_ok : setup_gn_test_64_g1.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_128_g1_ok : setup_gn_test_128_g1.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_512_g1_ok : setup_gn_test_512_g1.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_64_g4_ok : setup_gn_test_64_g4.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨4, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_128_g4_ok : setup_gn_test_128_g4.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨5, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_gn_test_1024_g4_ok : setup_gn_test_1024_g4.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨8, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts

/-! `LayerNorm` without a residual (`SETUP.md`): fla's `LayerNorm` test
(`tests/modules/test_layernorm.py:20-25`, 2 × 2 × 512 rows, `D` in 50, 64, 128, weight and bias),
`eps = 1e-5`, in `bfloat16` (the test runs `float32`). -/

def setup_ln_2048x50 : Inst := mkInst 2048 50 64 1 false eps1e5
def setup_ln_2048x64 : Inst := mkInst 2048 64 64 1 false eps1e5
def setup_ln_2048x128 : Inst := mkInst 2048 128 128 1 false eps1e5

theorem setup_ln_2048x50_ok : setup_ln_2048x50.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_ln_2048x64_ok : setup_ln_2048x64.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts
theorem setup_ln_2048x128_ok : setup_ln_2048x128.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts

end Claims.LayerNormFwd
