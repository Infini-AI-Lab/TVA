import TVA

/-!
# `layer_norm_gated_fwd_bf16`: gated RMSNorm forward (fla's `FusedRMSNormGated`)

The kernel is `layer_norm_gated_fwd_kernel` of `source/fused_norm_gate.py` (lines 55–127; it reads
the module-level names `triton`, its decorator, and `tl`, bound by the imports of lines 51–52).
Its host `layer_norm_gated_fwd` (lines 131–200) launches it on the grid `(triton.cdiv(T, BT),)`
for rows of `D ≤ 512`. The claim covers the configuration fla's linear-attention layers run
through `FusedRMSNormGated` (`SETUP.md`): RMSNorm (`IS_RMS_NORM = True`), a weight, no bias, no
residual, and the gate activation `"swish"` (or its alias `"silu"`) or `"sigmoid"`; `x`, `g`,
`weight` and `y` `bfloat16`, `rstd` `float32`; every `BT` of fla's autotune list (any power of
two with `BT·BD ≤ 2^20`).

What is claimed: every covered launch ends normally, and for each row `t < T`, with `x_j`, `g_j`,
`w_j` the row's initial `bfloat16` inputs and, exact in the reals with the host's double `eps`,
`rstdRef = 1 / sqrt(Σ_j x_j² / D + eps)`, `zRef j = x_j·rstdRef`, `σ(v) = 1 / (1 + e^-v)`, the gate
`gateRef j = g_j·σ(g_j)` (swish) or `σ(g_j)` (sigmoid) and `yRef j = zRef j · w_j · gateRef j`:
`rstd[t]` is a finite `float32` within `2^-12` (relative) of `rstdRef`; each `y[t, j]`, `j < D`,
is a finite `bfloat16` within `yTol j = 2^-7·|yRef j| + 2^-126·(1 + |zRef j|)·(1 + |w_j|)·gMag j`
of `yRef j`, with `gMag j = 1 + |g_j|` (swish) or `1` (sigmoid); every byte outside the elements of
`y` and `rstd` the launch writes is unchanged.

The conditions: `1 ≤ D ≤ 512` with `BD` a power of two, `D ≤ BD ≤ 512`; `BT` a power of two with
`BT·BD ≤ 2^20`; fewer than `2^31` programs; the buffers hold the `T·D` (`x`, `g`, `y`), `D`
(`weight`) and `T` (`rstd`) elements the launch touches, and the written elements share no byte
with the read ones or with each other; `eps` a finite double in `[2^-125, 2^126]`; every input a
finite `bfloat16`, `|g|, |weight| ≤ 2^56`, and every row's `Σ_j x[t, j]² + eps ≤ 2^125`.

The tolerances rest on the device facts for `tl.sqrt`, `tl.exp` and `/` on `float32`
(`Device.SqrtF32`, `Device.ExpF32`, `Device.ExpF32Overflow`, `Device.DivFull`,
`Device.DivF32InfDivisor`, and their totality facts).
-/

namespace Claims.LayerNormGatedFwd

open TVA

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `layer_norm_gated_fwd_kernel` (source lines 55–127), on the covered configuration. `x`, `g`,
`y`, `w` are `*bf16` and `rstd` is `*fp32`, the dtypes of the covered launches; `b`, `residual`,
`residual_out` and `mean` are `None` on every covered launch (no bias, no residual, and RMSNorm
allocates no `mean`): compile-time `None`s, which the body never reads on the branches the
covered flags take, so the model has no parameter for them. `eps` is a Python `float` argument: a
runtime `float32` scalar, `FloatV.hostF32` of the host's double (`model`). `T` is an unannotated
integer argument listed in `do_not_specialize`, so it is never the compile-time constant `1`: a
runtime `int32` for `T < 2^31` and a runtime `int64` from `2^31`; the model takes it as either
(`Sum.inl`, `Sum.inr`), as `model` picks by its value. `D`, `BT`, `BD`, `NB` are
`tl.constexpr`s, Lean parameters; the body never reads `NB` (an autotune key only). `ACTIVATION`
is a `tl.constexpr` string, `swish = true` for `"swish"` and `"silu"` and `false` for
`"sigmoid"`; the flags `IS_RMS_NORM = True`, `STORE_RESIDUAL_OUT = False`,
`HAS_RESIDUAL = False`, `HAS_WEIGHT = True`, `HAS_BIAS = False` are fixed by the covered
configuration; the branches they do not take are not modelled. -/
def layer_norm_gated_fwd_kernel (x g y w : Ptr .bf16) (rstd : Ptr .f32) (eps : F32)
    (T : Sum I32 I64) (D BT BD NB : Nat) (swish : Bool) : Kernel where
  body ctx := do
    -- line 79: `i_t = tl.program_id(0).to(tl.int64)`: axis 0 of the grid, an `int32` scalar,
    -- sign-extended to `int64`
    let i_t : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 81: `i_t * BT + tl.arange(0, BT)`: the constexpr `BT` built in `int64` (the dtype of
    -- the scalar tensor `i_t`), the `int64` product; the `int32` block `tl.arange(0, BT)`
    -- sign-extended to `int64` (the addition's promotion), the `int64` sum
    let o_a : Tensor [BT] I64 ← tl.convert .i32 .i64 (tl.arange 0 BT)
    let o_t : Tensor [BT] I64 := i_t * (IntV.ofNat BT : I64) + o_a
    -- line 82: an `int32` block
    let o_d : Tensor [BD] I32 := tl.arange 0 BD
    -- line 83: `o_d < D`: the constexpr `D` built in `int32` (the tensor's dtype), the signed
    -- comparison
    let m_d : Tensor [BD] Bool := o_d.ltS (IntV.ofNat D : I32)
    -- line 84: `m_t = o_t < T`: the scalar `T` promoted to `int64` (a sign extension when it is an
    -- `int32`; nothing when it is already an `int64`), then the signed comparison
    let T64 : I64 ← match T with
      | .inl t => do pure (← tl.convert .i32 .i64 (Tensor.scalar t)).item
      | .inr t => pure t
    let m_t : Tensor [BT] Bool := o_t.ltS T64
    -- line 85: `m_t[:, None] & m_d[None, :]`: the `[BT, 1]` and `[1, BD]` blocks broadcast to
    -- `[BT, BD]`; the model broadcasts the operands (`.col`, `.row`) and takes `&` lane by lane,
    -- the same lane values
    let m_x : Tensor [BT, BD] Bool := (m_t.col BD).and (m_d.row BT)
    -- line 87: `x + o_t[:, None] * D + o_d[None, :]`, evaluated left to right: the `int64` block
    -- `o_t[:, None]` times the constexpr `D` (built in `int64`), added to the `*bf16` pointer, then
    -- the `int32` offsets `o_d[None, :]` added to those pointers (each pointer advanced by its
    -- lane's offset, the `[BT, 1]` and `[1, BD]` operands broadcast to `[BT, BD]` first, the same
    -- lane values)
    let p_x : Tensor [BT, BD] (Ptr .bf16) := x + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT
    -- line 88: the masked load with `other=0.0`: the Python `0.0` is a `float32` tensor (`+0`),
    -- converted to the pointer's dtype, `+0` in `bfloat16` (exact; the value `litBF16` of `0`
    -- gives); then `.to(tl.float32)`, the exact widening (`tl.cvt`)
    let b_x16 : Tensor [BT, BD] BF16 ←
      tl.load p_x (some m_x) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
    let b_x : Tensor [BT, BD] F32 ← tl.cvt b_x16
    -- lines 89–94: `HAS_RESIDUAL` and `STORE_RESIDUAL_OUT` are `False`: not visited
    -- line 95: `not IS_RMS_NORM` is `False`: lines 96–100 are not visited, the `else` branch is
    -- line 102: `tl.where(m_d[None, :], b_x, 0.0)`: the `[1, BD]` condition broadcast to the rows;
    -- the Python `0.0`, a value of `tl.where`, built in the `float32` of `b_x` (`+0`)
    let b_xbar : Tensor [BT, BD] F32 :=
      tl.where (m_d.row BT) b_x (bc (FloatV.litF32 (FloatV.ofPat 0)))
    -- line 103: `tl.sum(b_xbar * b_xbar, axis=1) / D`. `b_xbar * b_xbar`: the IEEE `float32`
    -- product. `tl.sum(…, 1)`: the sum of products along axis 1; with `enable_fp_fusion` its
    -- additions may be contracted with the products: the fusable `tl.sum1c`, along a tree Triton
    -- chooses. `/ D`: the constexpr `D`, a Python `int` beside a `float32` tensor, built in
    -- `float32` (`litF32` of the double `D`, exact), and the platform's division (`tl.div`,
    -- `div.full.f32`)
    let b_xx : Tensor [BT, BD] F32 ← tl.mul b_xbar b_xbar
    let s1 : Tensor [BT] F32 ← tl.sum1c b_xx
    let b_var : Tensor [BT] F32 ← tl.div s1 (bc (FloatV.litF32 (FloatV.ofRat (D : ℚ))))
    -- line 104: `b_rstd = 1 / tl.sqrt(b_var + eps)`. `b_var + eps`: the runtime `float32` scalar
    -- broadcast to the rows; `b_var` is a `float32` quotient, a product for the fusion rule (and at
    -- `D = 1` MLIR folds `/ 1.0` away, leaving the sum of products), so the fusable `tl.addc`.
    -- `tl.sqrt(…)`: the platform's `float32` square root (`P.math .sqrt`, `sqrt.approx.ftz.f32`).
    -- `1 / …`: the Python `1`, an operand of `/`, built in the operation's dtype `float32` (the
    -- double `1.0`, `litF32`), and the platform's division (`tl.div`, `div.full.f32`)
    let s2 : Tensor [BT] F32 ← tl.addc b_var (bc eps)
    let sq : Tensor [BT] F32 ← tl.sqrt s2
    let b_rstd : Tensor [BT] F32 ←
      tl.div (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))) sq
    -- line 106: the `*fp32` pointer advanced by the `int64` block `o_t`
    let p_rstd : Tensor [BT] (Ptr .f32) := rstd + o_t
    -- line 107: `b_rstd.to(p_rstd.dtype.element_ty)`: `b_rstd` is already `float32`, the
    -- pointer's dtype, so no conversion; the masked store
    tl.store p_rstd b_rstd (some m_t)
    -- line 109: `HAS_WEIGHT` is `True`, the branch is visited
    -- line 110: `tl.load(w + o_d, mask=m_d)`: the `*bf16` pointer advanced by the `int32` block;
    -- the mask is `m_d` and there is no `other`: the lanes `d ≥ D` hold unspecified values, which
    -- no stored lane uses; then the exact widening
    let b_w16 : Tensor [BD] BF16 ← tl.load (w + o_d) (some m_d)
    let b_w : Tensor [BD] F32 ← tl.cvt b_w16
    -- lines 111–112: `HAS_BIAS` is `False`: not visited
    -- line 113: `IS_RMS_NORM` is `True`: the compile-time conditional evaluates its `else` side,
    -- `b_x * b_rstd[:, None]`, the IEEE `float32` product (`b_rstd` broadcast to the columns)
    let b_x_hat : Tensor [BT, BD] F32 ← tl.mul b_x (b_rstd.col BD)
    -- line 114: `HAS_WEIGHT` is `True`: `b_x_hat * b_w[None, :]`, the IEEE product (`b_w`
    -- broadcast to the rows)
    let b_y0 : Tensor [BT, BD] F32 ← tl.mul b_x_hat (b_w.row BT)
    -- lines 115–116: `HAS_BIAS` is `False`: not visited
    -- line 119: the `g` pointers, as line 87
    let p_g : Tensor [BT, BD] (Ptr .bf16) := g + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT
    -- line 120: the masked load with `other=0.0`, as line 88, then the exact widening
    let b_g16 : Tensor [BT, BD] BF16 ←
      tl.load p_g (some m_x) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
    let b_g : Tensor [BT, BD] F32 ← tl.cvt b_g16
    -- lines 121–124: a compile-time `if` on the string `ACTIVATION`
    let b_y : Tensor [BT, BD] F32 ← if swish then do
        -- line 122 (`"swish"`, `"silu"`): `b_y * b_g * tl.sigmoid(b_g)`, left to right: the IEEE
        -- product `b_y * b_g`, then `tl.sigmoid(b_g)` (Triton's body `1 / (1 + exp(-x))`,
        -- `tl.sigmoid`), then the IEEE product of the two
        let b_yg ← tl.mul b_y0 b_g
        let b_s ← tl.sigmoid b_g
        tl.mul b_yg b_s
      else do
        -- line 124 (`"sigmoid"`): `b_y * tl.sigmoid(b_g)`, the IEEE product
        let b_s ← tl.sigmoid b_g
        tl.mul b_y0 b_s
    -- line 126: the `y` pointers, as line 87
    let p_y : Tensor [BT, BD] (Ptr .bf16) := y + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT
    -- line 127: `b_y.to(p_y.dtype.element_ty)`: to `bfloat16`, one rounding to nearest even
    -- (`tl.cvt`); then the masked store
    let b_y16 : Tensor [BT, BD] BF16 ← tl.cvt b_y
    tl.store p_y b_y16 (some m_x)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `layer_norm_gated_fwd_kernel` through
`layer_norm_gated_fwd` (source lines 131–200) on the covered configuration. The tensors are
contiguous (`x`, `g`, `y` are `[T, D]`, `weight` `[D]`, `rstd` `[T]`), given by their storage. -/
structure Inst where
  /-- `T`: the rows (fla's `LayerNormGatedFunction` reshapes its input to `[-1, D]`: batch ×
  tokens × heads). -/
  T : Nat
  /-- `D`: the row length (`x.shape[-1]`). -/
  D : Nat
  /-- `BD`: the host's `min(65536 // 2, triton.next_power_of_2(D))` (line 162). -/
  BD : Nat
  /-- `NB`: the host's `triton.cdiv(T, 2048 * 32)`, an autotune key the body never reads. -/
  NB : Nat
  /-- `BT`: the rows per program (the host passes `16`; fla's autotune list has `16`, `32`, `64`). -/
  BT : Nat
  /-- `eps`: the host's Python `float` (a double); the kernel gets `FloatV.hostF32 eps`. -/
  eps : F64
  /-- `ACTIVATION`: `true` for `"swish"` and `"silu"`, `false` for `"sigmoid"`. -/
  swish : Bool
  /-- `x`: the input, `bfloat16`; `x[t, j]` is element `t·D + j`. -/
  x : Buf .bf16
  /-- `g`: the gate, `bfloat16`, laid out as `x`. -/
  g : Buf .bf16
  /-- `weight`: `bfloat16`, `D` elements. -/
  w : Buf .bf16
  /-- `y` (`torch.empty_like(x)`, line 153): the output, `bfloat16`, laid out as `x`. -/
  y : Buf .bf16
  /-- `rstd` (`torch.empty((T,), dtype=torch.float)`, line 159): `float32`, element `t` for row
  `t`. -/
  rstd : Buf .f32

/-- Element `(t, j)` of a contiguous `[T, D]` tensor. -/
def idx (d : Inst) (t j : Nat) : Nat := t * d.D + j

/-- The elements of a `[T, D]` tensor: the first `T·D`. -/
def matSet (d : Inst) : Set Nat := {k | k < d.T * d.D}

/-- The elements of `weight` the launch reads: the first `D`. -/
def vecSet (d : Inst) : Set Nat := {k | k < d.D}

/-- The elements of `rstd` the launch writes: the first `T`. -/
def rowSet (d : Inst) : Set Nat := {k | k < d.T}

/-- The bytes the launch reads. -/
def inBytes (d : Inst) : Set ℤ :=
  d.x.elems (matSet d) ∪ d.g.elems (matSet d) ∪ d.w.elems (vecSet d)

/-- The bytes the launch writes. -/
def outBytes (d : Inst) : Set ℤ := d.y.elems (matSet d) ∪ d.rstd.elems (rowSet d)

/-- The instances covered: every `D` of this path (`1 … 512`), every `BD` Triton accepts with
`D ≤ BD ≤ 512` (the host's `next_power_of_2(D)` among them), every power-of-two `BT` of a block
Triton accepts (the host's `16`, and every `BT` of fla's autotune list), every `T` whose grid CUDA
accepts, both activations, and every `eps` in `[2^-125, 2^126]`. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `D ≥ 1`: a tensor's last dimension, on this path. -/
  d_pos : 0 < d.D
  /-- `D ≤ 512`: this kernel's path (the host asserts it; fla launches another kernel beyond). -/
  d_le : d.D ≤ 512
  /-- `BD` is a power of two (Triton requires it of `tl.arange`). -/
  bd_pow2 : ∃ e, d.BD = 2 ^ e
  /-- `D ≤ BD`: the row fits in the block (the host's `BD = next_power_of_2(D)`). -/
  d_bd : d.D ≤ d.BD
  /-- `BD ≤ 512`: the host's `BD` for `D ≤ 512`. -/
  bd_le : d.BD ≤ 512
  /-- `BT` is a power of two (Triton requires it of `tl.arange`). -/
  bt_pow2 : ∃ e, d.BT = 2 ^ e
  /-- The block `[BT, BD]` has at most `2^20` elements (Triton's largest block). -/
  block_le : d.BT * d.BD ≤ 2 ^ 20
  /-- The grid `cdiv(T, BT)` has fewer than `2^31` programs (CUDA's limit on axis 0; a larger grid
  is rejected at launch). With `BT ≤ 2^20` it also gives `T < 2^51`, inside `int64`. -/
  grid_lt : (d.T + d.BT - 1) / d.BT < 2 ^ 31
  /-- `x` holds the `T·D` elements the launch reads. -/
  x_len : d.T * d.D ≤ d.x.len
  /-- `g` holds the `T·D` elements the launch reads. -/
  g_len : d.T * d.D ≤ d.g.len
  /-- `weight` holds the `D` elements the launch reads (the host's assert, line 149). -/
  w_len : d.D ≤ d.w.len
  /-- `y` holds the `T·D` elements the launch writes. -/
  y_len : d.T * d.D ≤ d.y.len
  /-- `rstd` holds the `T` elements the launch writes. -/
  r_len : d.T ≤ d.rstd.len
  /-- The written elements share no byte with the read ones (the host allocates `y` and `rstd`
  fresh). -/
  out_in : Disjoint (outBytes d) (inBytes d)
  /-- The written elements of `y` and of `rstd` share no byte. -/
  y_rstd : Disjoint (d.y.elems (matSet d)) (d.rstd.elems (rowSet d))
  /-- `eps` is a finite double. -/
  eps_fin : d.eps.isFinite = true
  /-- `eps ≥ 2^-125`: its `float32` is normal, and so is `var + eps` on every row, whatever the
  fusable addition gives, so it is in `tl.sqrt`'s domain (`sqrtDom`). Below `2^-126` the kernel is
  wrong on a zero row (README, Exclusions); real use has `eps ≥ 1e-6`. -/
  eps_lo : (2 : ℚ) ^ (-125 : ℤ) ≤ d.eps.toRat
  /-- `eps ≤ 2^126`. -/
  eps_hi : d.eps.toRat ≤ (2 : ℚ) ^ (126 : ℤ)

/-- The model on an instance: the buffers' first elements, `eps` as the `float32` the device gets
(`FloatV.hostF32` of the double), `T` as the runtime scalar Triton passes (`int32` below `2^31`,
`int64` from `2^31`), the constexprs and the activation. -/
def model (d : Inst) : Kernel :=
  layer_norm_gated_fwd_kernel (d.x.ptr 0) (d.g.ptr 0) (d.y.ptr 0) (d.w.ptr 0) (d.rstd.ptr 0)
    (FloatV.hostF32 d.eps)
    (if d.T < 2 ^ 31 then Sum.inl (IntV.ofNat d.T) else Sum.inr (IntV.ofNat d.T))
    d.D d.BT d.BD d.NB d.swish

/-- The host's launch (source lines 170–198): the grid `(triton.cdiv(T, BT),)`,
`(T + BT − 1) / BT`. The source's block shapes: `[BT]` (`tl.arange(0, BT)`, `o_t`, `m_t`,
`p_rstd`, `s1`, `b_var`, `s2`, `sq`, `b_rstd`), `[BD]` (`o_d`, `m_d`, the `weight` pointers,
`b_w`), `[BT, 1]` (`o_t[:, None]`, `m_t[:, None]`, the products with `D`, `b_rstd[:, None]`),
`[1, BD]` (`o_d[None, :]`, `m_d[None, :]`, `b_w[None, :]`) and `[BT, BD]` (`m_x`, the pointers
of `x`, `g`, `y`, the loads, the arithmetic, `b_y16`). The launch passes `num_warps = 4` and no
other option, so `enable_fp_fusion` is Triton's default `true`, `debug` is false, `num_ctas = 1`,
no `launch_pdl`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.T + d.BT - 1) / d.BT)
    [[d.BT], [d.BD], [d.BT, 1], [1, d.BD], [d.BT, d.BD]] (fusion := true)

/-- The `bfloat16` element `i` of a buffer in memory `m` (`+0` where absent; `Pre` makes every
element the launch reads present). -/
def rval (m : Mem) (p : Buf .bf16) (i : Nat) : BF16 :=
  match m.read .bf16 (p.ptr (i : Int)).addr with
  | some v => v
  | none => FloatV.ofPat 0

/-- `x[t, j]`, `g[t, j]` and `weight[j]` in the initial memory. -/
def xv (d : Inst) (init : Mem) (t j : Nat) : BF16 := rval init d.x (idx d t j)
def gv (d : Inst) (init : Mem) (t j : Nat) : BF16 := rval init d.g (idx d t j)
def wv (d : Inst) (init : Mem) (j : Nat) : BF16 := rval init d.w j

/-- A finite `bfloat16` of magnitude at most `B`. -/
def InOkB (B : ℚ) (a : BF16) : Prop := a.isFinite = true ∧ |a.toRat| ≤ B

/-- The exact sum of squares of row `t`, `Σ_{j<D} x[t, j]²`, in the reals. -/
noncomputable def rowSum (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  ∑ j ∈ Finset.range d.D, (xv d init t j).toReal ^ 2

/-- The inputs covered: the five buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); every element of `x` the launch reads is a finite `bfloat16`, every element of `g` and
of `weight` a finite `bfloat16` of magnitude at most `2^56` (README, Exclusions: far larger
products overflow); and on every row, `Σ_j x[t, j]² + eps ≤ 2^125` (the row's norm below `2^62.5`;
README, Exclusions). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.g.Valid init ∧ d.w.Valid init ∧ d.y.Valid init ∧ d.rstd.Valid init ∧
    (∀ t, t < d.T → ∀ j, j < d.D →
      (xv d init t j).isFinite = true ∧ InOkB (2 ^ 56) (gv d init t j)) ∧
    (∀ j, j < d.D → InOkB (2 ^ 56) (wv d init j)) ∧
    ∀ t, t < d.T → rowSum d init t + d.eps.toReal ≤ 2 ^ 125

/-! ## The specification

For row `t`, from the host's inputs, exact in the reals: `rstdRef = 1 / sqrt(Σ_j x_j² / D + eps)`
(the host's double `eps`), the normalised value `zRef j = x_j · rstdRef`, the logistic function
`σ(v) = 1 / (1 + e^-v)`, the gate `gateRef j` (`g_j·σ(g_j)` for swish, `σ(g_j)` for sigmoid) and
the output `yRef j = zRef j · w_j · gateRef j`. The tolerances (README, Tolerances, for their
derivation): `rstd` within `2^-12·rstdRef`; `y[t, j]` within
`2^-7·|yRef j| + 2^-126·(1 + |zRef j|)·(1 + |w_j|)·gMag j`, `gMag j = 1 + |g_j|` for swish and
`1` for sigmoid: the relative part is twice the output's `bfloat16` rounding, the floor covers the
`float32` underflow of the products and a gate `σ(g_j)` that `tl.exp`'s overflow makes `0` (at
`g_j ≤ -88.72`, where `σ(g_j) < 2^-128`). -/

/-- The exact factor of row `t`: `1 / sqrt(Σ_j x_j² / D + eps)`, with the host's `eps`. -/
noncomputable def rstdRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (Real.sqrt (rowSum d init t / d.D + d.eps.toReal))⁻¹

/-- The exact normalised value `x[t, j] · rstdRef`. -/
noncomputable def zRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  (xv d init t j).toReal * rstdRef d init t

/-- The logistic function `σ(v) = 1 / (1 + e^-v)`. -/
noncomputable def sig (v : ℝ) : ℝ := 1 / (1 + Real.exp (-v))

/-- The exact gate: `g·σ(g)` for swish, `σ(g)` for sigmoid (lines 121–124). -/
noncomputable def gateRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  if d.swish then (gv d init t j).toReal * sig (gv d init t j).toReal
  else sig (gv d init t j).toReal

/-- The exact output `zRef · weight[j] · gateRef`. -/
noncomputable def yRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  zRef d init t j * (wv d init j).toReal * gateRef d init t j

/-- The gate's magnitude in the floor of `yTol`: `1 + |g|` for swish, `1` for sigmoid. -/
noncomputable def gMag (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  if d.swish then 1 + |(gv d init t j).toReal| else 1

/-- The relative tolerance of `rstd[t]`, `2^-12` (README, Tolerances). -/
def rstdTol : ℚ := 2 ^ (-12 : ℤ)

/-- The tolerance of `y[t, j]` (README, Tolerances). -/
noncomputable def yTol (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  2 ^ (-7 : ℤ) * |yRef d init t j| +
    2 ^ (-126 : ℤ) * (1 + |zRef d init t j|) * (1 + |(wv d init j).toReal|) * gMag d init t j

/-- What the launch leaves in row `t < T`: `rstd[t]` is a finite `float32` within `rstdTol`
(relative) of `rstdRef`, and each `y[t, j]`, `j < D`, is a finite `bfloat16` within `yTol j` of
`yRef j`. -/
def RowDone (d : Inst) (init fin : Mem) (t : Nat) : Prop :=
  (∃ c : F32, fin.read .f32 (d.rstd.ptr (t : Int)).addr = some c ∧ c.isFinite = true ∧
    |c.toReal - rstdRef d init t| ≤ (rstdTol : ℝ) * rstdRef d init t) ∧
  ∀ j, j < d.D → ∃ c : BF16, fin.read .bf16 (d.y.ptr (idx d t j : Int)).addr = some c ∧
    c.isFinite = true ∧ |c.toReal - yRef d init t j| ≤ yTol d init t j

/-- What the launch leaves: every row as `RowDone` says, and every byte outside the elements of
`y` and `rstd` the launch writes (`outBytes`) unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.T → RowDone d init fin t) ∧ ∀ z, z ∉ outBytes d → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `layer_norm_gated_fwd_kernel` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The double `1e-6` (the `norm_eps` of fla's GatedDeltaNet, GLA and KDA configurations),
pattern `0x3EB0C6F7A0B5ED8D`. -/
def eps1e6 : F64 := FloatV.ofPat 0x3EB0C6F7A0B5ED8D

/-- The double `1e-5` (`FusedRMSNormGated`'s default `eps`), pattern `0x3EE4F8B588E368F1`. -/
def eps1e5 : F64 := FloatV.ofPat 0x3EE4F8B588E368F1

theorem eps1e6_facts : eps1e6.isFinite = true ∧ (2 : ℚ) ^ (-125 : ℤ) ≤ eps1e6.toRat ∧
    eps1e6.toRat ≤ (2 : ℚ) ^ (126 : ℤ) ∧ eps1e6.toRat ≤ 1 := by
  refine ⟨by decide +kernel, ?_, ?_, ?_⟩
  · have : (2 ^ (-125 : ℤ) : ℚ) ≤ (FloatV.ofPat 0x3EB0C6F7A0B5ED8D : F64).toRat := by
      decide +kernel
    exact this
  · have : (FloatV.ofPat 0x3EB0C6F7A0B5ED8D : F64).toRat ≤ (2 ^ (126 : ℤ) : ℚ) := by
      decide +kernel
    exact this
  · have : (FloatV.ofPat 0x3EB0C6F7A0B5ED8D : F64).toRat ≤ 1 := by decide +kernel
    exact this

theorem eps1e5_facts : eps1e5.isFinite = true ∧ (2 : ℚ) ^ (-125 : ℤ) ≤ eps1e5.toRat ∧
    eps1e5.toRat ≤ (2 : ℚ) ^ (126 : ℤ) ∧ eps1e5.toRat ≤ 1 := by
  refine ⟨by decide +kernel, ?_, ?_, ?_⟩
  · have : (2 ^ (-125 : ℤ) : ℚ) ≤ (FloatV.ofPat 0x3EE4F8B588E368F1 : F64).toRat := by
      decide +kernel
    exact this
  · have : (FloatV.ofPat 0x3EE4F8B588E368F1 : F64).toRat ≤ (2 ^ (126 : ℤ) : ℚ) := by
      decide +kernel
    exact this
  · have : (FloatV.ofPat 0x3EE4F8B588E368F1 : F64).toRat ≤ 1 := by decide +kernel
    exact this

/-- A launch of the host on `T` rows of `D` elements (`BD = next_power_of_2(D)`): `BT = 16`,
`NB = cdiv(T, 2048·32)`, the activation `sw` (swish when `true`) and the double `e` as `eps`; the
five tensors at `2^36`, `2·2^36`, … (each well below `2^36` bytes). -/
def mkInst (T D BD : Nat) (sw : Bool) (e : F64) : Inst where
  T := T
  D := D
  BD := BD
  NB := (T + 65535) / 65536
  BT := 16
  eps := e
  swish := sw
  x := ⟨2 ^ 36, T * D⟩
  g := ⟨2 * 2 ^ 36, T * D⟩
  w := ⟨3 * 2 ^ 36, D⟩
  y := ⟨4 * 2 ^ 36, T * D⟩
  rstd := ⟨5 * 2 ^ 36, T⟩

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

theorem mkInst_ok {T D BD : Nat} {sw : Bool} {e : F64} (hd : 1 ≤ D) (hd' : D ≤ 512)
    (hbd : ∃ k, BD = 2 ^ k) (hdbd : D ≤ BD) (hbd' : BD ≤ 512) (htd : T * D ≤ 2 ^ 30)
    (he : e.isFinite = true ∧ (2 : ℚ) ^ (-125 : ℤ) ≤ e.toRat ∧ e.toRat ≤ (2 : ℚ) ^ (126 : ℤ) ∧
      e.toRat ≤ 1) :
    (mkInst T D BD sw e).ok := by
  have hT : T ≤ 2 ^ 30 := le_trans (Nat.le_mul_of_pos_right T hd) htd
  have hTZ : ((T * D : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast htd
  have hTZ' : ((T : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hT
  have hDZ : ((D : Nat) : ℤ) ≤ 512 := by exact_mod_cast hd'
  push_cast at hTZ
  have mat : ∀ j ∈ matSet (mkInst T D BD sw e), j < T * D := fun j hj => hj
  have row : ∀ j ∈ rowSet (mkInst T D BD sw e), j < T := fun j hj => hj
  have vec : ∀ j ∈ vecSet (mkInst T D BD sw e), j < D := fun j hj => hj
  refine ⟨hd, hd', hbd, hdbd, hbd', ⟨4, rfl⟩, ?_, ?_, le_rfl, le_rfl, le_rfl, le_rfl, le_rfl,
    ?_, ?_, he.1, he.2.1, he.2.2.1⟩
  · show 16 * BD ≤ 2 ^ 20
    omega
  · show (T + 16 - 1) / 16 < 2 ^ 31
    omega
  · simp only [outBytes, inBytes]
    refine Disjoint.union_left ?_ ?_ <;>
      refine Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_ <;>
      refine elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) ?_ ?_ <;>
      first | exact mat | exact row | exact vec
  · exact elems_disj _ _ (by simp only [mkInst, DType.bytes]; push_cast; omega) mat row

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

/-- Every aligned `bfloat16` read of `zeroMem` gives `+0`. -/
theorem zeroMem_read16 {a : Int} (ha : Mem.Aligned .bf16 a) :
    zeroMem.read .bf16 a = some (FloatV.ofPat 0 : BF16) := by
  have hb : zeroMem.readBytes a 2 = some [0, 0] := by simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  rw [show DType.bf16.bytes = 2 from rfl]
  simp only [ha, ite_true, hb]
  decide

/-- An element of a buffer of `mkInst`, inside its first `2^36` bytes, reads `+0` in `zeroMem`. -/
theorem zeroMem_rval {p : Buf .bf16} {i : Nat} (h1 : 0 ≤ (p.ptr (i : Int)).addr)
    (h2 : (p.ptr (i : Int)).addr + 2 ≤ 2 ^ 64) (h3 : (p.ptr (i : Int)).addr % 2 = 0) :
    rval zeroMem p i = FloatV.ofPat 0 := by
  unfold rval
  rw [zeroMem_read16 ⟨h1, h2, h3⟩]

/-- `+0` is a finite `bfloat16` of value `0`. -/
theorem zero_facts : (FloatV.ofPat 0 : BF16).isFinite = true ∧ (FloatV.ofPat 0 : BF16).toRat = 0 :=
  ⟨by decide, by decide +kernel⟩

theorem mkInst_pre {T D BD : Nat} {sw : Bool} {e : F64} (htd : T * D ≤ 2 ^ 30) (hT : T ≤ 2 ^ 30)
    (hD : D ≤ 512) (he : e.toRat ≤ 1) : Pre (mkInst T D BD sw e) zeroMem := by
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid zeroMem :=
    fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [zeroMem]⟩
  have hTZ : ((T * D : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast htd
  have hTZ' : ((T : Nat) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hT
  have hDZ : ((D : Nat) : ℤ) ≤ 512 := by exact_mod_cast hD
  push_cast at hTZ
  have hidx : ∀ t j, t < T → j < D → t * D + j < T * D := fun t j ht hj =>
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left hj _)
      (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ ht)
  have hx0 : ∀ t j, t < T → j < D → xv (mkInst T D BD sw e) zeroMem t j = FloatV.ofPat 0 := by
    intro t j ht hj
    have h1 : ((t * D + j : Nat) : ℤ) < ((T * D : Nat) : ℤ) := by exact_mod_cast hidx t j ht hj
    push_cast at h1
    have h0 : (0 : ℤ) ≤ (t : ℤ) * (D : ℤ) := by positivity
    apply zeroMem_rval <;> simp only [Buf.ptr, mkInst, idx, DType.bytes] <;> push_cast <;> omega
  refine ⟨hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    hv _ (by simp [mkInst]) (by simp only [mkInst, DType.bytes]; push_cast; omega)
      (by simp [mkInst, DType.bytes]),
    fun t ht j hj => ?_, fun j hj => ?_, fun t ht => ?_⟩
  · change t < T at ht
    change j < D at hj
    have h1 : ((t * D + j : Nat) : ℤ) < ((T * D : Nat) : ℤ) := by exact_mod_cast hidx t j ht hj
    push_cast at h1
    have h0 : (0 : ℤ) ≤ (t : ℤ) * (D : ℤ) := by positivity
    have hg0 : gv (mkInst T D BD sw e) zeroMem t j = FloatV.ofPat 0 := by
      apply zeroMem_rval <;> simp only [Buf.ptr, mkInst, idx, DType.bytes] <;> push_cast <;> omega
    rw [hx0 t j ht hj, hg0, InOkB, zero_facts.2]
    exact ⟨zero_facts.1, zero_facts.1, by norm_num⟩
  · change j < D at hj
    have hjZ : ((j : Nat) : ℤ) < (D : ℤ) := by exact_mod_cast hj
    have hw0 : wv (mkInst T D BD sw e) zeroMem j = FloatV.ofPat 0 := by
      apply zeroMem_rval <;> simp only [Buf.ptr, mkInst, DType.bytes] <;> push_cast <;> omega
    rw [hw0, InOkB, zero_facts.2]
    exact ⟨zero_facts.1, by norm_num⟩
  · change t < T at ht
    have hsum : rowSum (mkInst T D BD sw e) zeroMem t = 0 := by
      unfold rowSum
      refine Finset.sum_eq_zero fun j hj => ?_
      have hj' : j < D := Finset.mem_range.mp hj
      rw [hx0 t j ht hj']
      simp [FloatV.toReal, zero_facts.2]
    rw [hsum]
    have he' : (mkInst T D BD sw e).eps.toReal ≤ 1 := by
      show ((e.toRat : ℚ) : ℝ) ≤ 1
      exact_mod_cast he
    have h1 : (1 : ℝ) ≤ 2 ^ 125 := one_le_pow₀ (by norm_num)
    linarith

theorem claim_satisfiable : claim.Satisfiable :=
  ⟨mkInst 1 1 1 true eps1e5, zeroMem,
    mkInst_ok (by norm_num) (by norm_num) ⟨0, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
      eps1e5_facts,
    mkInst_pre (by norm_num) (by norm_num) (by norm_num) eps1e5_facts.2.2.2⟩

/-! ## Coverage

The host's test launches (source lines 211–239, `(T, D, activation, eps)`; the first three are
fla's RMSNorm-gated test shapes on this path, `tests/modules/test_layernorm_gated.py:72-80`,
there with `eps = 0`, here with `1e-5`). -/

def setup_4x64_silu : Inst := mkInst 4 64 64 true eps1e5
def setup_2048x128_sigmoid : Inst := mkInst 2048 128 128 false eps1e5
def setup_200x50_sigmoid : Inst := mkInst 200 50 64 false eps1e5
def setup_98304x512_swish : Inst := mkInst 98304 512 512 true eps1e6
def setup_262144x128_sigmoid : Inst := mkInst 262144 128 128 false eps1e5
def setup_1x1_swish : Inst := mkInst 1 1 1 true eps1e5
def setup_33x300_swish : Inst := mkInst 33 300 512 true eps1e5

theorem setup_4x64_silu_ok : setup_4x64_silu.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e5_facts
theorem setup_2048x128_sigmoid_ok : setup_2048x128_sigmoid.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e5_facts
theorem setup_200x50_sigmoid_ok : setup_200x50_sigmoid.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨6, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e5_facts
theorem setup_98304x512_swish_ok : setup_98304x512_swish.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e6_facts
theorem setup_262144x128_sigmoid_ok : setup_262144x128_sigmoid.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e5_facts
theorem setup_1x1_swish_ok : setup_1x1_swish.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨0, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e5_facts
theorem setup_33x300_swish_ok : setup_33x300_swish.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e5_facts

/-! The layers of `SETUP.md`, on a training batch of 4 × 4096 tokens and on one decoding token of
a batch of 8 (one row per token and head, `eps = 1e-6`, their configurations' `norm_eps`):
GatedDeltaNet's output norm (6 heads of 512, swish), GLA's (4 heads of 512, swish) and KDA's
(16 heads of 128, sigmoid). -/

def setup_gdn_train : Inst := mkInst 98304 512 512 true eps1e6
def setup_gdn_decode : Inst := mkInst 48 512 512 true eps1e6
def setup_gla_train : Inst := mkInst 65536 512 512 true eps1e6
def setup_gla_decode : Inst := mkInst 32 512 512 true eps1e6
def setup_kda_train : Inst := mkInst 262144 128 128 false eps1e6
def setup_kda_decode : Inst := mkInst 128 128 128 false eps1e6

theorem setup_gdn_train_ok : setup_gdn_train.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e6_facts
theorem setup_gdn_decode_ok : setup_gdn_decode.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e6_facts
theorem setup_gla_train_ok : setup_gla_train.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e6_facts
theorem setup_gla_decode_ok : setup_gla_decode.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e6_facts
theorem setup_kda_train_ok : setup_kda_train.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e6_facts
theorem setup_kda_decode_ok : setup_kda_decode.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    eps1e6_facts

end Claims.LayerNormGatedFwd
