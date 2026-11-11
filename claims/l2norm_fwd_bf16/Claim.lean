import TVA

/-!
# `l2norm_fwd_bf16`: L2 normalisation of rows, forward (fla)

The kernel is `l2norm_fwd_kernel` of `source/l2norm.py` (lines 44–60; it reads the module-level
names `triton`, its decorator, and `tl`, bound by the imports of lines 40–41). Its host
`l2norm_fwd` (lines 63–90) launches it on the grid `(triton.cdiv(T, 16),)` with `BT = 16` and
`num_warps = 4`, on a contiguous `bfloat16` `x` viewed as `[T, D]`, a fresh `y` laid out as `x`
(`bfloat16` for `output_dtype = None`, `float32` for `output_dtype = torch.float32`) and a fresh
`float32` `rstd` of `T` elements; `SETUP.md` lists the launches. Each row `t < T` is divided by its
L2 norm, computed in `float32` over the padded row length `BD`.

What is claimed: every covered launch ends normally, and for each row `t < T`, with `a_j` the row's
initial `bfloat16` inputs and `rstdRef = 1 / sqrt(Σ_j a_j² + eps)` exact in the reals with the
host's double `eps`: `rstd[t]` is a finite `float32` within `rstdTol = 2^-12` (relative) of
`rstdRef`; each `y[t, j]`, `j < D`, is a finite value of `y`'s dtype within
`yTol·|a_j|·rstdRef + yFloor` of `a_j·rstdRef` (`bfloat16`: `2^-7`, `2^-132`; `float32`: `yTol32 =
2^-11`, `yFloor32 = 2^-149`); every byte outside the elements of `y` and `rstd` the launch writes is
unchanged.

The conditions: `1 ≤ D ≤ 512` with `BD` a power of two, `D ≤ BD ≤ 512`; `BT` a power of two with
`BT·BD ≤ 2^20`; fewer than `2^31` programs; the buffers hold the `T·D` (`x`, `y`) and `T` (`rstd`)
elements the launch touches, and the written elements of `y` and `rstd` share no byte with each
other or with the read elements of `x`; `eps` a finite double in `[2^-125, 2^126]`; every input a
finite `bfloat16`, and every row's `Σ_j x[t, j]² + eps ≤ 2^126`.

The tolerances rest on the device facts for `tl.sqrt` and `/` on `float32` (`Device.SqrtF32`,
`Device.DivFull`, and their totality facts).
-/

namespace Claims.L2NormFwd

open TVA

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `l2norm_fwd_kernel` (source lines 44–60). `x` is `*bf16`, `rstd` is `*fp32`, and `y` is `*bf16`
(`Sum.inl`) or `*fp32` (`Sum.inr`): the host passes a `bfloat16` `x`, `y = empty_like(x)` or
`empty_like(x, dtype=output_dtype)` with `output_dtype = torch.float32`, and a `float32` `rstd`;
the pointer's dtype is part of Triton's specialisation, so each is its own compiled kernel, and
the model takes the two as the two cases of `y`. `eps` is a Python `float`
argument: a runtime `float32` scalar, `FloatV.hostF32` of the host's double (`model`). `T` is an
unannotated integer argument listed in `do_not_specialize`, so it is never the compile-time constant
`1`: a runtime `int32` for `T < 2^31` and a runtime `int64` from `2^31`; the model takes it as
either (`Sum.inl`, `Sum.inr`), as `model` picks by its value.
`D`, `BD`, `NB` and `BT` are `tl.constexpr`s, Lean parameters; the body never reads `NB` (an
autotune key only). -/
def l2norm_fwd_kernel (x : Ptr .bf16) (y : Sum (Ptr .bf16) (Ptr .f32)) (rstd : Ptr .f32)
    (eps : F32) (T : Sum I32 I64) (D BD NB BT : Nat) : Kernel where
  body ctx := do
    -- line 46: `i_t = tl.program_id(0).to(tl.int64)`: axis 0 of the grid, an `int32` scalar,
    -- sign-extended to `int64`
    let i_t : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 47: `i_t * BT + tl.arange(0, BT)`: the constexpr `BT` built in `int64` (the dtype of
    -- the scalar tensor `i_t`), the `int64` product; the `int32` block `tl.arange(0, BT)`
    -- sign-extended to `int64` (the addition's promotion), the `int64` sum
    let o_a : Tensor [BT] I64 ← tl.convert .i32 .i64 (tl.arange 0 BT)
    let o_t : Tensor [BT] I64 := i_t * (IntV.ofNat BT : I64) + o_a
    -- line 48: an `int32` block
    let o_d : Tensor [BD] I32 := tl.arange 0 BD
    -- line 49: `m_t = o_t < T`: the scalar `T` promoted to `int64` (a sign extension when it is an
    -- `int32`; nothing when it is already an `int64`), then the signed comparison
    let T64 : I64 ← match T with
      | .inl t => do pure (← tl.convert .i32 .i64 (Tensor.scalar t)).item
      | .inr t => pure t
    let m_t : Tensor [BT] Bool := o_t.ltS T64
    -- line 50: `m_t[:, None] & (o_d[None, :] < D)`: `o_d[None, :]` against the constexpr `D`
    -- built in `int32` (the tensor's dtype), then the `&` of the `[BT, 1]` and `[1, BD]` blocks,
    -- broadcast to `[BT, BD]`; the model broadcasts the operands (`.col`, `.row`) and takes `&`
    -- lane by lane, the same lane values
    let m_d : Tensor [BD] Bool := o_d.ltS (IntV.ofNat D : I32)
    let m_x : Tensor [BT, BD] Bool := (m_t.col BD).and (m_d.row BT)
    -- line 51: `x + o_t[:, None] * D + o_d[None, :]`, evaluated left to right: the `int64` block
    -- `o_t[:, None]` times the constexpr `D` (built in `int64`), added to the `*bf16` pointer, then
    -- the `int32` offsets `o_d[None, :]` added to those pointers (each pointer advanced by its
    -- lane's offset, the `[BT, 1]` and `[1, BD]` operands broadcast to `[BT, BD]` first, the same
    -- lane values)
    let p_x : Tensor [BT, BD] (Ptr .bf16) :=
      x + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT
    -- line 52: the same pointers into `y`, of its dtype
    let p_y : Sum (Tensor [BT, BD] (Ptr .bf16)) (Tensor [BT, BD] (Ptr .f32)) := match y with
      | .inl y => .inl (y + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
      | .inr y => .inr (y + o_t.col BD * (IntV.ofNat D : I64) + o_d.row BT)
    -- line 53: the `*fp32` pointer advanced by the `int64` block `o_t`
    let p_rstd : Tensor [BT] (Ptr .f32) := rstd + o_t
    -- line 55: the masked load with `other=0.0`: the Python `0.0` is a `float32` tensor (`+0`),
    -- converted to the pointer's dtype, `+0` in `bfloat16` (exact; the value `litBF16` of `0`
    -- gives); then `.to(tl.float32)`, the exact widening (`tl.cvt`)
    let b_x16 : Tensor [BT, BD] BF16 ←
      tl.load p_x (some m_x) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
    let b_x : Tensor [BT, BD] F32 ← tl.cvt b_x16
    -- line 56: `b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)`.
    -- `b_x * b_x`: the IEEE `float32` product.
    -- `tl.sum(…, 1)`: the sum of products along axis 1; with `enable_fp_fusion` its additions may
    -- be contracted with the products: the fusable `tl.sum1c`, along a tree Triton chooses.
    -- `… + eps`: the runtime `float32` scalar broadcast to the rows. A product feeds this addition
    -- through the sum (at `BD = 1` the sum is the product itself, and the backend may contract
    -- `x*x + eps` into one `fma`), so the fusable `tl.addc`.
    -- `tl.sqrt(…)`: the platform's `float32` square root (`P.math .sqrt`, `sqrt.approx.ftz.f32`).
    -- `1 / …`: the Python `1`, an operand of `/`, built in the operation's dtype `float32` (the
    -- double `1.0`, `litF32`), and the platform's division (`tl.div`, `div.full.f32`).
    let b_xx : Tensor [BT, BD] F32 ← tl.mul b_x b_x
    let s1 : Tensor [BT] F32 ← tl.sum1c b_xx
    let s2 : Tensor [BT] F32 ← tl.addc s1 (bc eps)
    let sq : Tensor [BT] F32 ← tl.sqrt s2
    let b_rstd : Tensor [BT] F32 ←
      tl.div (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))) sq
    -- line 57: `b_x * b_rstd[:, None]`: `b_rstd` broadcast to the columns, the IEEE `float32`
    -- product (no addition follows it)
    let b_y : Tensor [BT, BD] F32 ← tl.mul b_x (b_rstd.col BD)
    -- line 59: `b_y.to(p_y.dtype.element_ty)`: to `bfloat16`, one rounding to nearest even
    -- (`tl.cvt`); to `float32`, `b_y`'s own dtype, `b_y` itself; then the masked store
    match p_y with
    | .inl p_y => do
      let b_y16 : Tensor [BT, BD] BF16 ← tl.cvt b_y
      tl.store p_y b_y16 (some m_x)
    | .inr p_y => tl.store p_y b_y (some m_x)
    -- line 60: `b_rstd.to(p_rstd.dtype.element_ty)`: `b_rstd` is already `float32`, the pointer's
    -- dtype, so no conversion; the masked store
    tl.store p_rstd b_rstd (some m_t)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `l2norm_fwd_kernel`. -/
structure Inst where
  /-- `T`: the rows of the view `x.view(-1, x.shape[-1])` (batch × tokens × heads). -/
  T : Nat
  /-- `D`: `x.shape[-1]`, the row length. -/
  D : Nat
  /-- `BD`: the host's `min(65536 // 2, triton.next_power_of_2(D))`. -/
  BD : Nat
  /-- `NB`: the host's `triton.cdiv(T, 2048 * 32)`, an autotune key the body never reads. -/
  NB : Nat
  /-- `BT`: the rows per program (the host passes `16`). -/
  BT : Nat
  /-- `eps`: the host's Python `float` (a double); the kernel gets `FloatV.hostF32 eps`. -/
  eps : F64
  /-- `x`: the input, `bfloat16`, contiguous; `x[t, j]` is element `t·D + j`. -/
  x : Buf .bf16
  /-- `y`: the output, laid out as `x`: `bfloat16` (`Sum.inl`, `output_dtype = None`) or `float32`
  (`Sum.inr`, `output_dtype = torch.float32`). -/
  y : Sum (Buf .bf16) (Buf .f32)
  /-- `rstd`: the output factors, `float32`, element `t` for row `t`. -/
  rstd : Buf .f32

/-- Element `j` of row `t` of the `[T, D]` view: `t·D + j`. -/
def xIdx (d : Inst) (t j : Nat) : Nat := t * d.D + j

/-- The elements of `x` the launch reads and of `y` it writes: the first `T·D`. -/
def rowSet (d : Inst) : Set Nat := {k | k < d.T * d.D}

/-- The elements of `rstd` the launch writes: the first `T`. -/
def rSet (d : Inst) : Set Nat := {t | t < d.T}

/-- The bytes of the elements `S` of `y`, of either dtype. -/
def yElems (d : Inst) (S : Set Nat) : Set Int :=
  match d.y with
  | .inl b => b.elems S
  | .inr b => b.elems S

/-- The number of elements of `y`. -/
def yLen (d : Inst) : Nat :=
  match d.y with
  | .inl b => b.len
  | .inr b => b.len

/-- `y`'s buffer is valid in memory `m`. -/
def yValid (d : Inst) (m : Mem) : Prop :=
  match d.y with
  | .inl b => b.Valid m
  | .inr b => b.Valid m

/-- `y`'s first element, the kernel's argument. -/
def yPtr (d : Inst) : Sum (Ptr .bf16) (Ptr .f32) :=
  match d.y with
  | .inl b => .inl (b.ptr 0)
  | .inr b => .inr (b.ptr 0)

/-- The instances covered: every `D` of this path (`1 … 512`), every `BD` Triton accepts with
`D ≤ BD ≤ 512` (the host's `next_power_of_2(D)` among them), every power-of-two `BT` of a block
Triton accepts (the host's `16`, and every `BT` of fla's autotune list), every `T` whose grid CUDA
accepts, and every `eps` in `[2^-125, 2^126]`. -/
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
  /-- `y` holds the `T·D` elements the launch writes. -/
  y_len : d.T * d.D ≤ yLen d
  /-- `rstd` holds the `T` elements the launch writes. -/
  r_len : d.T ≤ d.rstd.len
  /-- The elements of `y` the launch writes share no byte with the elements of `x` it reads (the
  host allocates `y` fresh). -/
  y_x : Disjoint (yElems d (rowSet d)) (d.x.elems (rowSet d))
  /-- The elements of `rstd` the launch writes share no byte with the elements of `x` it reads. -/
  r_x : Disjoint (d.rstd.elems (rSet d)) (d.x.elems (rowSet d))
  /-- The elements of `rstd` the launch writes share no byte with the elements of `y` it writes. -/
  r_y : Disjoint (d.rstd.elems (rSet d)) (yElems d (rowSet d))
  /-- `eps` is a finite double. -/
  eps_fin : d.eps.isFinite = true
  /-- `eps ≥ 2^-125`: its `float32` is normal, and so is `sum + eps` on every row, whatever the
  fusable addition gives, so it is in `tl.sqrt`'s domain (`sqrtDom`). Below `2^-126` the kernel is
  wrong on a zero row (README, Exclusions); real use has `eps ≥ 1e-12`. -/
  eps_lo : (2 : ℚ) ^ (-125 : ℤ) ≤ d.eps.toRat
  /-- `eps ≤ 2^126` (with `Pre`'s row bound, `sum + eps` stays inside `float32`). -/
  eps_hi : d.eps.toRat ≤ (2 : ℚ) ^ (126 : ℤ)

/-- The model on an instance: the buffers' first elements, `eps` as the `float32` the device gets
(`FloatV.hostF32` of the double), `T` as the runtime scalar Triton passes (`int32` below `2^31`,
`int64` from `2^31`), and the constexprs. -/
def model (d : Inst) : Kernel :=
  l2norm_fwd_kernel (d.x.ptr 0) (yPtr d) (d.rstd.ptr 0) (FloatV.hostF32 d.eps)
    (if d.T < 2 ^ 31 then Sum.inl (IntV.ofNat d.T) else Sum.inr (IntV.ofNat d.T))
    d.D d.BD d.NB d.BT

/-- The host's launch (source line 89): the grid `(triton.cdiv(T, BT),)`, `(T + BT − 1) / BT`. The
source's block shapes: `[BT]` (`tl.arange(0, BT)`, `o_t`, `m_t`, `p_rstd`, `s1`, `s2`, `sq`,
`b_rstd`), `[BD]` (`o_d`), `[BT, 1]` (`o_t[:, None]`, `m_t[:, None]`, the product with `D` and its
pointers), `[1, BD]` (`o_d[None, :]` and its comparison) and `[BT, BD]` (`m_x`, `p_x`, `p_y`,
`b_x16`, `b_x`, `b_xx`, `b_y`, and `b_y16` for a `bfloat16` `y`). The launch passes `num_warps = 4` and no other option, so
`enable_fp_fusion` is Triton's default `true`, `debug` is false, `num_ctas = 1`, no `launch_pdl`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.T + d.BT - 1) / d.BT)
    [[d.BT], [d.BD], [d.BT, 1], [1, d.BD], [d.BT, d.BD]] (fusion := true)

/-- The exact sum of squares of a row, `Σ_{j<D} a_j²`, in the reals. -/
noncomputable def rowSum (a : Nat → BF16) (D : Nat) : ℝ := ∑ j ∈ Finset.range D, (a j).toReal ^ 2

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); every element the launch reads is a finite `bfloat16`; and on every row,
`Σ_j x[t, j]² + eps ≤ 2^126` (the row's norm below `2^63`; README, Exclusions). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ yValid d init ∧ d.rstd.Valid init ∧
    (∀ t, t < d.T → ∀ j, j < d.D → ∃ a : BF16,
      init.read .bf16 (d.x.ptr (xIdx d t j)).addr = some a ∧ a.isFinite = true) ∧
    ∀ t, t < d.T → ∀ a : Nat → BF16,
      (∀ j, j < d.D → init.read .bf16 (d.x.ptr (xIdx d t j)).addr = some (a j)) →
        rowSum a d.D + d.eps.toReal ≤ 2 ^ 126

/-! ## The specification -/

/-- The exact factor of a row: `1 / sqrt(Σ_j a_j² + eps)`, in the reals, with the host's `eps`. -/
noncomputable def rstdRef (a : Nat → BF16) (D : Nat) (e : ℝ) : ℝ := (√(rowSum a D + e))⁻¹

/-- The relative tolerance of `rstd[t]`, `2^-12` (README, Tolerances). -/
def rstdTol : ℚ := 2 ^ (-12 : ℤ)

/-- The relative tolerance of `y[t, j]`, `2^-7`, relative to `|x[t, j]|·rstdRef` (README,
Tolerances). -/
def yTol : ℚ := 2 ^ (-7 : ℤ)

/-- The absolute floor of `y[t, j]`, `2^-132` (README, Tolerances). -/
def yFloor : ℚ := 2 ^ (-132 : ℤ)

/-- The relative tolerance of a `float32` `y[t, j]`, `2^-11`, relative to `|x[t, j]|·rstdRef`
(README, Tolerances). -/
def yTol32 : ℚ := 2 ^ (-11 : ℤ)

/-- The absolute floor of a `float32` `y[t, j]`, `2^-149` (README, Tolerances). -/
def yFloor32 : ℚ := 2 ^ (-149 : ℤ)

/-- Element `k` of `y` in the final memory is a finite value of `y`'s dtype within the tolerance
of that dtype of `ref` (an output `a·rstdRef`, with `scale = |a|·rstdRef`): `yTol·scale + yFloor`
for `bfloat16`, `yTol32·scale + yFloor32` for `float32`. -/
def YOk (d : Inst) (fin : Mem) (k : Nat) (ref scale : ℝ) : Prop :=
  match d.y with
  | .inl b => ∃ c : BF16, fin.read .bf16 (b.ptr k).addr = some c ∧ c.isFinite = true ∧
      |c.toReal - ref| ≤ (yTol : ℝ) * scale + (yFloor : ℝ)
  | .inr b => ∃ c : F32, fin.read .f32 (b.ptr k).addr = some c ∧ c.isFinite = true ∧
      |c.toReal - ref| ≤ (yTol32 : ℝ) * scale + (yFloor32 : ℝ)

/-- What the launch leaves: for each row `t < T`, with `a j` the row's `bfloat16` inputs in the
initial memory, `rstd[t]` is a finite `float32` within `rstdTol` (relative) of the exact
`1 / sqrt(Σ_j a_j² + eps)`, and each `y[t, j]`, `j < D`, is a finite value of `y`'s dtype within
`yTol·|a_j|·rstdRef + yFloor` (`bfloat16`) or `yTol32·|a_j|·rstdRef + yFloor32` (`float32`) of the
exact `a_j · rstdRef` (`YOk`); every byte outside the elements of `y` and `rstd` the launch writes
is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.T → ∃ a : Nat → BF16,
      (∀ j, j < d.D → init.read .bf16 (d.x.ptr (xIdx d t j)).addr = some (a j)) ∧
      (∃ rc : F32, fin.read .f32 (d.rstd.ptr t).addr = some rc ∧ rc.isFinite = true ∧
        |rc.toReal - rstdRef a d.D d.eps.toReal| ≤
          (rstdTol : ℝ) * rstdRef a d.D d.eps.toReal) ∧
      ∀ j, j < d.D → YOk d fin (xIdx d t j) ((a j).toReal * rstdRef a d.D d.eps.toReal)
        (|(a j).toReal| * rstdRef a d.D d.eps.toReal)) ∧
    ∀ z, z ∉ yElems d (rowSet d) ∪ d.rstd.elems (rSet d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `l2norm_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The double `1e-6` (fla's default `eps`), pattern `0x3EB0C6F7A0B5ED8D`. -/
def eps1e6 : F64 := FloatV.ofPat 0x3EB0C6F7A0B5ED8D

theorem eps1e6_fin : eps1e6.isFinite = true := by decide

theorem eps1e6_lo : (2 : ℚ) ^ (-125 : ℤ) ≤ eps1e6.toRat := by decide +kernel

theorem eps1e6_hi : eps1e6.toRat ≤ (2 : ℚ) ^ (126 : ℤ) := by decide +kernel

/-- A launch of the host on `T` rows of `D` elements (`BD = next_power_of_2(D)`): `BT = 16`,
`NB = cdiv(T, 2048·32)`, `eps = 1e-6`, `x`, `y`, `rstd` at the byte addresses `2^30`, `2^31`,
`2^31 + 2^30`; `y` is `bfloat16` (`output_dtype = None`), or `float32` when `f32`
(`output_dtype = torch.float32`). -/
def mk (T D BD : Nat) (f32 : Bool := false) : Inst where
  T := T
  D := D
  BD := BD
  NB := (T + 65535) / 65536
  BT := 16
  eps := eps1e6
  x := ⟨2 ^ 30, T * D⟩
  y := if f32 then .inr ⟨2 ^ 31, T * D⟩ else .inl ⟨2 ^ 31, T * D⟩
  rstd := ⟨2 ^ 31 + 2 ^ 30, T⟩

theorem bytes_disjoint {d₁ d₂ : DType} (b : Buf d₁) (c : Buf d₂)
    (h : b.base + b.len * d₁.bytes ≤ c.base ∨ c.base + c.len * d₂.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro z hz hz'
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hz hz'
  omega

theorem mk_ok (T D BD e : Nat) (hD : 0 < D) (hD' : D ≤ 512) (he : BD = 2 ^ e) (hDB : D ≤ BD)
    (hB : BD ≤ 512) (hT : T * D ≤ 2 ^ 28) (hT' : T ≤ 2 ^ 27) (f32 : Bool := false) :
    (mk T D BD f32).ok := by
  cases f32 <;>
  refine ⟨hD, hD', ⟨e, he⟩, hDB, hB, ⟨4, rfl⟩, ?_, ?_, le_refl _, le_refl _, le_refl _,
    ?_, ?_, ?_, eps1e6_fin, eps1e6_lo, eps1e6_hi⟩ <;>
  first
    | (show 16 * BD ≤ 2 ^ 20; omega)
    | (show (T + 16 - 1) / 16 < 2 ^ 31; omega)
    | (refine Buf.elems_disjoint_of _ _ (bytes_disjoint _ _ ?_) (fun j hj => hj) (fun j hj => hj)
       right
       simp only [mk, DType.bytes]
       push_cast
       omega)

/-! ## Coverage: the host's test cases (source lines 105–115, `(B, T, H, D)`, `T = B·T·H` rows)
and the configurations SETUP.md names -/

/-- `(1, 1, 1, 1)`: one row, `D = 1`. -/
def setup_1_1_1_1 : Inst := mk 1 1 1
theorem setup_1_1_1_1_ok : setup_1_1_1_1.ok := mk_ok 1 1 1 0 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1, 1, 1, 128)`. -/
def setup_1_1_1_128 : Inst := mk 1 128 128
theorem setup_1_1_1_128_ok : setup_1_1_1_128.ok := mk_ok 1 128 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1, 63, 1, 60)`: `BD = 64`. -/
def setup_1_63_1_60 : Inst := mk 63 60 64
theorem setup_1_63_1_60_ok : setup_1_63_1_60.ok := mk_ok 63 60 64 6 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1, 65, 2, 512)`: `D = 512`, the largest on this path. -/
def setup_1_65_2_512 : Inst := mk 130 512 512
theorem setup_1_65_2_512_ok : setup_1_65_2_512.ok := mk_ok 130 512 512 9 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(2, 500, 4, 64)`. -/
def setup_2_500_4_64 : Inst := mk 4000 64 64
theorem setup_2_500_4_64_ok : setup_2_500_4_64.ok := mk_ok 4000 64 64 6 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(2, 1000, 2, 100)`: `BD = 128`. -/
def setup_2_1000_2_100 : Inst := mk 4000 100 128
theorem setup_2_1000_2_100_ok : setup_2_1000_2_100.ok := mk_ok 4000 100 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(3, 1024, 4, 128)`. -/
def setup_3_1024_4_128 : Inst := mk 12288 128 128
theorem setup_3_1024_4_128_ok : setup_3_1024_4_128.ok := mk_ok 12288 128 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(2, 333, 3, 256)`. -/
def setup_2_333_3_256 : Inst := mk 1998 256 256
theorem setup_2_333_3_256_ok : setup_2_333_3_256.ok := mk_ok 1998 256 256 8 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(4, 4096, 16, 128)`: DeltaNet's 16 heads of 128 on a training batch of 4 × 4096 tokens
(SETUP.md), `2^18` rows. -/
def setup_4_4096_16_128 : Inst := mk 262144 128 128
theorem setup_4_4096_16_128_ok : setup_4_4096_16_128.ok := mk_ok 262144 128 128 7 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- GatedDeltaNet's 6 heads of 256 on a training batch of 4 × 4096 tokens (SETUP.md). -/
def setup_gdn_4_4096_6_256 : Inst := mk 98304 256 256
theorem setup_gdn_4_4096_6_256_ok : setup_gdn_4_4096_6_256.ok := mk_ok 98304 256 256 8 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- PaTH attention's `w = l2_norm(w, output_dtype=torch.float32)` (SETUP.md): 32 heads of 64 on a
training batch of 4 × 4096 tokens, a `float32` `y`. -/
def setup_path_4_4096_32_64_f32 : Inst := mk 524288 64 64 true
theorem setup_path_4_4096_32_64_f32_ok : setup_path_4_4096_32_64_f32.ok := mk_ok 524288 64 64 6 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) true

/-- The same on one decoding token of a batch of 8: `8 × 32` rows, a `float32` `y`. -/
def setup_path_8_1_32_64_f32 : Inst := mk 256 64 64 true
theorem setup_path_8_1_32_64_f32_ok : setup_path_8_1_32_64_f32.ok := mk_ok 256 64 64 6 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) true

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

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨setup_1_1_1_1, zeroMem, setup_1_1_1_1_ok, ?_, ?_, ?_, ?_, ?_⟩
  · exact ⟨by norm_num [setup_1_1_1_1, mk], by norm_num [setup_1_1_1_1, mk, DType.bytes],
      by norm_num [setup_1_1_1_1, mk, DType.bytes], fun _ _ => by simp [zeroMem]⟩
  · exact ⟨by norm_num [setup_1_1_1_1, mk], by norm_num [setup_1_1_1_1, mk, DType.bytes],
      by norm_num [setup_1_1_1_1, mk, DType.bytes], fun _ _ => by simp [zeroMem]⟩
  · exact ⟨by norm_num [setup_1_1_1_1, mk], by norm_num [setup_1_1_1_1, mk, DType.bytes],
      by norm_num [setup_1_1_1_1, mk, DType.bytes], fun _ _ => by simp [zeroMem]⟩
  · intro t ht j hj
    have ht0 : t = 0 := by simp [setup_1_1_1_1, mk] at ht; omega
    have hj0 : j = 0 := by simp [setup_1_1_1_1, mk] at hj; omega
    subst ht0; subst hj0
    refine ⟨FloatV.ofPat 0, zeroMem_read16 ?_, by decide⟩
    unfold Mem.Aligned
    simp [Buf.ptr_addr, xIdx, setup_1_1_1_1, mk, DType.bytes]
  · intro t ht a ha
    have ht0 : t = 0 := by simp [setup_1_1_1_1, mk] at ht; omega
    subst ht0
    have h0 : a 0 = FloatV.ofPat 0 := by
      have h := ha 0 (by simp [setup_1_1_1_1, mk])
      rw [zeroMem_read16 (by unfold Mem.Aligned; simp [Buf.ptr_addr, xIdx, setup_1_1_1_1, mk, DType.bytes])] at h
      exact (Option.some.inj h).symm
    have hz : (a 0).toReal = 0 := by
      rw [h0]; show ((Format.bfloat16.toRat 0 : ℚ) : ℝ) = 0
      have : Format.bfloat16.toRat 0 = 0 := by decide +kernel
      rw [this]; simp
    have he : eps1e6.toReal ≤ 1 := by
      have : eps1e6.toRat ≤ 1 := by decide +kernel
      show ((eps1e6.toRat : ℚ) : ℝ) ≤ 1
      exact_mod_cast this
    show rowSum a 1 + eps1e6.toReal ≤ 2 ^ 126
    simp only [rowSum, Finset.sum_range_one, hz]
    nlinarith

end Claims.L2NormFwd
